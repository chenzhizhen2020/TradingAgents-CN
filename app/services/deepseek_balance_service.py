"""
DeepSeek balance query service.

The API key is resolved and used only on the backend. Responses returned by this
service intentionally contain no key material.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from app.core.database import get_mongo_db
from app.services.config_service import config_service
from app.utils.api_key_utils import get_env_api_key_for_provider, is_valid_api_key

logger = logging.getLogger("app.services.deepseek_balance_service")


class DeepSeekBalanceService:
    """Fetch and normalize DeepSeek account balance snapshots."""

    endpoint = "https://api.deepseek.com/user/balance"

    async def _resolve_api_key(self) -> Optional[str]:
        try:
            db = get_mongo_db()
            provider = await db.llm_providers.find_one({"name": "deepseek"})
            if provider and is_valid_api_key(provider.get("api_key")):
                return provider["api_key"]
        except Exception as e:
            logger.warning("读取 DeepSeek 厂家 API Key 失败: %s", e)

        try:
            config = await config_service.get_system_config()
            for item in config.llm_configs if config else []:
                provider = item.provider.value if hasattr(item.provider, "value") else str(item.provider)
                if provider == "deepseek" and is_valid_api_key(item.api_key):
                    return item.api_key
        except Exception as e:
            logger.warning("读取 DeepSeek 模型 API Key 失败: %s", e)

        return get_env_api_key_for_provider("deepseek")

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _unavailable_snapshot(self, reason: str) -> Dict[str, Any]:
        return {
            "available": False,
            "is_available": False,
            "error": reason,
            "queried_at": datetime.utcnow().isoformat(),
            "balances_by_currency": {},
            "balance_infos": [],
        }

    async def get_balance_snapshot(self) -> Dict[str, Any]:
        api_key = await self._resolve_api_key()
        if not api_key:
            return self._unavailable_snapshot("未配置 DeepSeek API Key")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    self.endpoint,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as e:
            logger.warning("查询 DeepSeek 余额失败: %s", e)
            return self._unavailable_snapshot(f"查询 DeepSeek 余额失败: {e}")

        balance_infos = payload.get("balance_infos") or []
        balances_by_currency: Dict[str, float] = {}
        normalized_infos = []

        for item in balance_infos:
            currency = str(item.get("currency") or "UNKNOWN").upper()
            total_balance = self._to_float(item.get("total_balance"))
            balances_by_currency[currency] = balances_by_currency.get(currency, 0.0) + total_balance
            normalized_infos.append({
                "currency": currency,
                "total_balance": total_balance,
                "granted_balance": self._to_float(item.get("granted_balance")),
                "topped_up_balance": self._to_float(item.get("topped_up_balance")),
            })

        return {
            "available": bool(payload.get("is_available", False)),
            "is_available": bool(payload.get("is_available", False)),
            "queried_at": datetime.utcnow().isoformat(),
            "balances_by_currency": balances_by_currency,
            "balance_infos": normalized_infos,
        }


deepseek_balance_service = DeepSeekBalanceService()
