"""Phase 3A：Master 总并发、后台任务跟踪与失败回传。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.config.constants import AGENT_MASTER
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.modules.agent_cluster import master_agent as master_module


def _message(task_id: str) -> MCPMessage:
    return MCPMessage(
        task_id=task_id,
        sender="api_gateway",
        target=AGENT_MASTER,
        content={"task_type": "goods_search", "query": "bag"},
    )


def test_master_total_concurrency_limit_is_enforced():
    async def scenario():
        active = 0
        peak = 0
        entered_two = asyncio.Event()
        release = asyncio.Event()

        async def blocked_process(_msg, _memory):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered_two.set()
            await release.wait()
            active -= 1

        old_sem = master_module._master_task_semaphore
        master_module._master_task_semaphore = None
        try:
            with patch.object(master_module.settings, "MASTER_MAX_CONCURRENT", 2), patch.object(
                master_module, "process_master_task", new=AsyncMock(side_effect=blocked_process)
            ):
                tasks = [
                    asyncio.create_task(
                        master_module.safe_process_master_task(_message(f"root-{i}"), AsyncMock())
                    )
                    for i in range(3)
                ]
                await asyncio.wait_for(entered_two.wait(), timeout=1)
                await asyncio.sleep(0)
                assert active == 2
                release.set()
                await asyncio.gather(*tasks)
            return peak
        finally:
            master_module._master_task_semaphore = old_sem

    assert asyncio.run(scenario()) == 2


def test_background_tasks_are_tracked_and_removed_after_completion():
    async def scenario():
        release = asyncio.Event()

        async def blocked_safe(_msg, _memory):
            await release.wait()

        with patch.object(master_module, "safe_process_master_task", side_effect=blocked_safe):
            task = master_module._track_master_task(_message("tracked"), AsyncMock())
            await asyncio.sleep(0)
            assert task in master_module._master_tasks
            release.set()
            await task
            await asyncio.sleep(0)
            assert task not in master_module._master_tasks

    asyncio.run(scenario())


def test_unexpected_master_exception_returns_failure_reply():
    async def scenario():
        request = _message("failed-root")
        old_sem = master_module._master_task_semaphore
        master_module._master_task_semaphore = None
        try:
            with patch.object(
                master_module, "process_master_task", new=AsyncMock(side_effect=RuntimeError("secret"))
            ), patch.object(
                master_module.mcp_bus, "send_msg", new=AsyncMock(return_value=True)
            ) as send:
                await master_module.safe_process_master_task(request, AsyncMock())
            return send.await_args.args[0]
        finally:
            master_module._master_task_semaphore = old_sem

    reply = asyncio.run(scenario())
    assert reply.task_id == "failed-root"
    assert reply.content["success"] is False
    assert reply.content["error_msg"] == "master task failed"
    assert "secret" not in str(reply.content)


def test_background_exception_is_consumed_and_task_removed():
    async def scenario():
        async def exploding_safe(_msg, _memory):
            raise RuntimeError("background failure")

        with patch.object(master_module, "safe_process_master_task", side_effect=exploding_safe):
            task = master_module._track_master_task(_message("background"), AsyncMock())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert task.done()
            assert task not in master_module._master_tasks
            # done callback 已调用 task.exception()；再次读取仍安全且不会产生未处理告警。
            assert isinstance(task.exception(), RuntimeError)

    asyncio.run(scenario())
