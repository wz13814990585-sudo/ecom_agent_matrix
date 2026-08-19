"""总控规划 Agent（ReAct 多轮编排）。"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ecom_agent_matrix.config.constants import AGENT_MASTER
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import register_agent
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter, is_agent_reply
from ecom_agent_matrix.core.llm.output_polish import polish_final_output
from ecom_agent_matrix.modules.agent_cluster.master_planner import (
    merge_observation_into_working,
    plan_sub_tasks_llm,
    react_decide,
    should_save_to_memory,
)
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
import ecom_agent_matrix.modules.skills  # noqa: F401  # 确保 skill 已注册

logger = setup_logger("agent.master")

_subtask_semaphore: asyncio.Semaphore | None = None


def _get_subtask_semaphore() -> asyncio.Semaphore:
    global _subtask_semaphore
    if _subtask_semaphore is None:
        _subtask_semaphore = asyncio.Semaphore(settings.MASTER_MAX_SUBTASK_CONCURRENT)
    return _subtask_semaphore


def aggregate_sub_replies(task_id: str, replies: list[MCPMessage], expected: int) -> dict[str, Any]:
    """聚合子 Agent 回传结果。"""
    sub_results = []
    for msg in replies:
        sub_results.append(
            {
                "agent": msg.sender,
                "success": msg.content.get("success", False),
                "data": msg.content.get("data", {}),
                "error_msg": msg.content.get("error_msg", ""),
            }
        )
    return {
        "task_id": task_id,
        "expected": expected,
        "received": len(replies),
        "timed_out": len(replies) < expected,
        "all_success": bool(replies) and all(r["success"] for r in sub_results),
        "sub_results": sub_results,
    }


def _observation_from_reply(reply: MCPMessage | None, target_agent: str, timed_out: bool) -> dict:
    if reply is None:
        return {
            "agent": target_agent,
            "success": False,
            "data": {},
            "error_msg": "子任务超时无回传",
            "timed_out": True,
        }
    return {
        "agent": reply.sender or target_agent,
        "success": bool(reply.content.get("success")) and not timed_out,
        "data": reply.content.get("data") or {},
        "error_msg": reply.content.get("error_msg", ""),
        "timed_out": timed_out,
    }


async def _dispatch_subtask(task_id: str, target_agent: str, payload: dict, priority: int) -> None:
    """限流后向子 Agent 下发单步任务。"""
    sem = _get_subtask_semaphore()
    async with sem:
        # 去掉过大的内部字段；Master 不写业务解析
        clean = {k: v for k, v in payload.items() if k not in ("_memory_context",)}
        sub_msg = MCPMessage(
            task_id=task_id,
            sender=AGENT_MASTER,
            target=target_agent,
            priority=priority,
            content=clean,
        )
        await mcp_bus.send_msg(sub_msg)
        logger.info(
            "subtask_dispatched",
            extra={"event": "subtask_dispatched", "task_id": task_id, "agent": target_agent},
        )


async def _react_call_skill(skill_name: str, payload: dict) -> dict:
    """ReAct 单步：直接执行 Skill，返回 observation。"""
    clean = {k: v for k, v in payload.items() if k not in ("_memory_context",)}
    result = await exec_skill(skill_name, clean)
    return {
        "agent": skill_name,
        "kind": "skill",
        "success": bool(result.success),
        "data": result.data or {},
        "error_msg": result.error_msg or "",
        "timed_out": False,
    }


async def _react_call_one(
    task_id: str,
    target_agent: str,
    payload: dict,
    priority: int,
) -> dict:
    """ReAct 单步：下发一个 Agent → 等待回传 → 返回 observation。"""
    TaskReplyWaiter.begin(task_id, 1)
    await _dispatch_subtask(task_id, target_agent, payload, priority)
    replies = await TaskReplyWaiter.wait(task_id, timeout=float(settings.MCP_TIMEOUT))
    timed_out = len(replies) < 1
    reply = replies[0] if replies else None
    return _observation_from_reply(reply, target_agent, timed_out)


async def process_master_task(msg: MCPMessage, long_mem: AgentLongVectorMemory) -> None:
    """
    ReAct 主流程：
    召回 → 初始规划（建议序列）→ 逐步 Thought/Action/Observation → 聚合回传 → 条件记忆。
    子 Agent 仅 Query / Exec / RAG；SKU 解析在 Query 内部完成。
    """
    task_id = msg.task_id
    started = time.perf_counter()
    task_input = dict(msg.content or {})
    task_type = task_input.get("task_type", "unknown")

    logger.info(
        "master_task_received",
        extra={"event": "master_task_received", "task_id": task_id, "query": str(task_input)[:200]},
    )

    memory_hits: list[dict] = []
    try:
        memory_hits = await long_mem.recall(
            query_text=json.dumps(task_input, ensure_ascii=False),
            agent_name=AGENT_MASTER,
            top_k=2,
        )
    except Exception as exc:
        logger.warning(
            "master_memory_recall_failed",
            extra={"event": "master_memory_recall_failed", "task_id": task_id, "error": str(exc)},
        )

    plan = await plan_sub_tasks_llm(task_input, memory_hits)
    suggested_agents = [s["target_agent"] for s in plan.sub_tasks]

    logger.info(
        "master_plan_done",
        extra={
            "event": "master_plan_done",
            "task_id": task_id,
            "planner": plan.planner,
            "plan_confidence": plan.plan_confidence,
            "agents": suggested_agents,
        },
    )

    working = dict(plan.sub_tasks[0]["payload"]) if plan.sub_tasks else dict(task_input)
    observations: list[dict] = []
    react_trace: list[dict] = []
    max_steps = int(settings.MASTER_REACT_MAX_STEPS)

    for step in range(1, max_steps + 1):
        decision = await react_decide(working, observations, suggested_agents)
        react_trace.append(
            {
                "step": step,
                "thought": decision.thought,
                "action": decision.action,
                "agent": decision.agent,
                "skill": decision.skill,
                "source": decision.source,
                "reasoning_content": decision.reasoning_content or "",
            }
        )
        logger.info(
            "master_react_step",
            extra={
                "event": "master_react_step",
                "task_id": task_id,
                "query": (
                    f"step={step} action={decision.action} "
                    f"agent={decision.agent} skill={decision.skill}"
                ),
            },
        )

        if decision.action == "finish":
            react_trace[-1]["final_answer"] = decision.final_answer
            break

        if decision.action == "call_skill":
            if not decision.skill:
                react_trace[-1]["final_answer"] = "call_skill 缺少 skill 名，提前结束"
                break
            obs = await _react_call_skill(decision.skill, decision.payload)
            observations.append(obs)
            working = merge_observation_into_working(working, obs)
            react_trace[-1]["observation_success"] = obs.get("success")
            react_trace[-1]["error_msg"] = obs.get("error_msg", "")
            if not obs.get("success"):
                break
            continue

        if decision.action != "call_agent" or not decision.agent:
            react_trace[-1]["final_answer"] = "非法 ReAct 动作，提前结束"
            break

        obs = await _react_call_one(task_id, decision.agent, decision.payload, msg.priority)
        observations.append(obs)
        working = merge_observation_into_working(working, obs)
        react_trace[-1]["observation_success"] = obs.get("success")
        react_trace[-1]["error_msg"] = obs.get("error_msg", "")

        # 关键失败且后续依赖该结果时，规则层会在下一步 finish；此处也可提前跳出
        if obs.get("timed_out"):
            break

    else:
        react_trace.append(
            {
                "step": max_steps + 1,
                "thought": "达到 ReAct 最大步数",
                "action": "finish",
                "final_answer": "达到最大步数，强制结束",
            }
        )

    expected = max(len(observations), 1)
    final_result = {
        "task_id": task_id,
        "mode": "react",
        "expected": expected,
        "received": len(observations),
        "timed_out": any(o.get("timed_out") for o in observations) or not observations,
        "all_success": bool(observations) and all(o.get("success") for o in observations),
        "sub_results": [
            {
                "agent": o.get("agent"),
                "success": o.get("success", False),
                "data": o.get("data", {}),
                "error_msg": o.get("error_msg", ""),
            }
            for o in observations
        ],
        "react_trace": react_trace,
        "working_sku": working.get("sku"),
        "plan": {
            "planner": plan.planner,
            "plan_confidence": plan.plan_confidence,
            "reasoning": plan.reasoning,
            "agents": suggested_agents,
        },
    }
    elapsed_ms = (time.perf_counter() - started) * 1000

    summary = await polish_final_output(
        success=final_result["all_success"] and not final_result["timed_out"],
        data=final_result,
        error_msg=(
            "部分子任务超时或未成功"
            if final_result["timed_out"] or not final_result["all_success"]
            else ""
        ),
        user_query=str(task_input.get("query") or task_input.get("user_query") or ""),
        reply_from=AGENT_MASTER,
        prefer_existing_answer=False,
    )
    final_result["summary"] = summary

    reply = build_reply(
        msg,
        sender=AGENT_MASTER,
        success=final_result["all_success"] and not final_result["timed_out"],
        data=final_result,
        error_msg=(
            "部分子任务超时或未成功"
            if final_result["timed_out"] or not final_result["all_success"]
            else ""
        ),
        msg_type="master_task_result",
    )
    await mcp_bus.send_msg(reply)

    save_ok, mem_confidence = should_save_to_memory(plan.plan_confidence, final_result)
    if save_ok:
        await long_mem.safe_save_memory(
            agent_name=AGENT_MASTER,
            content=json.dumps(
                {
                    "task_id": task_id,
                    "input": task_input,
                    "plan": final_result["plan"],
                    "react_trace": react_trace,
                    "output": final_result,
                },
                ensure_ascii=False,
            ),
            meta={
                "task_type": task_type,
                "planner": plan.planner,
                "plan_confidence": plan.plan_confidence,
                "confidence": mem_confidence,
                "memory_hits_used": len(memory_hits),
                "latency_ms": round(elapsed_ms, 2),
                "mode": "react",
                "sku": working.get("sku"),
                "success": True,
                "verified": False,
                "deprecated": False,
            },
        )
        logger.info(
            "master_memory_saved",
            extra={
                "event": "master_memory_saved",
                "task_id": task_id,
                "confidence": mem_confidence,
            },
        )
    else:
        logger.info(
            "master_memory_skipped",
            extra={
                "event": "master_memory_skipped",
                "task_id": task_id,
                "plan_confidence": plan.plan_confidence,
                "memory_confidence": mem_confidence,
                "all_success": final_result["all_success"],
                "timed_out": final_result["timed_out"],
            },
        )

    logger.info(
        "master_task_done",
        extra={
            "event": "master_task_done",
            "task_id": task_id,
            "recall_count": final_result["received"],
            "latency_ms": round(elapsed_ms, 2),
        },
    )


@register_agent(AGENT_MASTER)
async def master_agent(msg_queue: asyncio.Queue):
    """总控 Master Agent 主循环。"""
    logger.info("master_agent_started", extra={"event": "master_agent_started", "agent": AGENT_MASTER})
    long_mem = AgentLongVectorMemory()

    while True:
        msg: MCPMessage = await msg_queue.get()
        try:
            if is_agent_reply(msg):
                TaskReplyWaiter.submit_reply(msg)
                continue
            asyncio.create_task(process_master_task(msg, long_mem))
        except Exception as exc:
            logger.exception(
                "master_loop_error",
                extra={"event": "master_loop_error", "task_id": msg.task_id, "error": str(exc)},
            )
        finally:
            msg_queue.task_done()
