"""Agent 长向量记忆模块。"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.modules.rag.embedding import get_text_embedding
from ecom_agent_matrix.core.security import tenant_scope_from_task_context

logger = logging.getLogger("memory.long")


class AgentLongVectorMemory:
    """Agent 长期向量记忆：pgvector 存储历史决策，支持语义召回。"""

    TABLE = "agent_long_memory"

    @staticmethod
    def _trusted_scope(context: Any | None) -> dict[str, str]:
        trusted = bool(
            getattr(context, "authenticated", False)
            or getattr(context, "identity_trusted", False)
        )
        if not trusted:
            return {}
        tenant_id = str(getattr(context, "tenant_id", "") or "").strip()
        store_id = str(getattr(context, "store_id", "") or "").strip()
        if not tenant_id or not store_id:
            return {}
        return {"tenant_id": tenant_id, "store_id": store_id}

    async def save_memory(
        self,
        agent_name: str,
        content: str,
        meta: dict,
        *,
        context: Any | None = None,
    ) -> Optional[int]:
        trusted_scope = self._trusted_scope(context)
        scoped_meta = {**dict(meta or {}), **trusted_scope}
        vec = await get_text_embedding(content)
        scope = tenant_scope_from_task_context(context)
        if scope.usable:
            sql = f"""
            INSERT INTO {self.TABLE}(tenant_id, store_id, agent_name, content, embedding, meta_json)
            VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb) RETURNING id;
            """
            params = [scope.tenant_id, scope.store_id, agent_name, content, vec,
                      json.dumps(scoped_meta, ensure_ascii=False)]
        else:
            sql = f"""
            INSERT INTO {self.TABLE}(agent_name, content, embedding, meta_json)
            VALUES (%s, %s, %s::vector, %s::jsonb) RETURNING id;
            """
            params = [agent_name, content, vec, json.dumps(scoped_meta, ensure_ascii=False)]
        res = await AsyncPGClient.execute_write(sql, params, scope=scope)
        return res[0][0]

    async def safe_save_memory(
        self,
        agent_name: str,
        content: str,
        meta: dict,
        *,
        context: Any | None = None,
    ) -> Optional[int]:
        """写入长期记忆；失败只打日志，不阻断主流程。"""
        try:
            mem_id = await self.save_memory(agent_name, content, meta, context=context)
            logger.info(
                "long_memory_saved agent=%s id=%s confidence=%s",
                agent_name,
                mem_id,
                meta.get("confidence"),
            )
            return mem_id
        except Exception as exc:
            logger.warning(
                "long_memory_save_failed agent=%s error_type=%s",
                agent_name,
                type(exc).__name__,
            )
            return None

    async def recall(
        self,
        query_text: str,
        agent_name: str,
        top_k: int = 3,
        min_confidence: float | None = None,
        meta_filter: dict | None = None,
        *,
        context: Any | None = None,
    ) -> List[Dict]:
        """
        召回高质量记忆：排除 deprecated，优先 success + 高置信度。
        meta_filter 按 meta_json 精确过滤（如 {"sku": "SKU-BAG-001"}），避免仅靠向量相似度串 SKU。
        """
        min_conf = (
            min_confidence
            if min_confidence is not None
            else settings.MASTER_MEMORY_RECALL_MIN_CONFIDENCE
        )
        q_vec = await get_text_embedding(query_text)
        scope = tenant_scope_from_task_context(context)

        where_extra = ""
        params: list = [q_vec, agent_name, min_conf]
        enforced_filter = dict(meta_filter or {})
        enforced_filter.pop("tenant_id", None)
        enforced_filter.pop("store_id", None)
        if enforced_filter:
            for key, value in enforced_filter.items():
                where_extra += " AND meta_json->>%s = %s"
                params.extend([str(key), str(value)])

        params.append(top_k)
        sql = f"""
        SELECT id, content, meta_json, embedding <-> %s::vector AS dist
        FROM {self.TABLE}
        WHERE agent_name = %s
          AND COALESCE((meta_json->>'deprecated')::boolean, false) = false
          AND COALESCE((meta_json->>'success')::boolean, true) = true
          AND COALESCE((meta_json->>'confidence')::float, 1.0) >= %s
          {where_extra}
        ORDER BY dist ASC
        LIMIT %s;
        """
        if scope.usable:
            sql = sql.replace(
                "WHERE agent_name = %s",
                "WHERE agent_name = %s AND tenant_id = %s AND store_id = %s",
            )
            params[2:2] = [scope.tenant_id, scope.store_id]
        rows = await AsyncPGClient.execute_read(sql, params, scope=scope)
        return [
            {"id": r[0], "content": r[1], "meta": r[2], "distance": r[3]}
            for r in rows
        ]

    async def deprecate_memory(self, memory_id: int, *, context: Any | None = None) -> bool:
        """软删除错误记忆，防止再次召回。"""
        sql = f"""
        UPDATE {self.TABLE}
        SET meta_json = COALESCE(meta_json, '{{}}'::jsonb) || '{{"deprecated": true}}'::jsonb
        WHERE id = %s
        """
        scope = tenant_scope_from_task_context(context)
        params: list = [memory_id]
        if scope.usable:
            sql += " AND tenant_id = %s AND store_id = %s"
            params.extend([scope.tenant_id, scope.store_id])
        sql += " RETURNING id;"
        rows = await AsyncPGClient.execute_write(sql, params, scope=scope)
        return bool(rows)
