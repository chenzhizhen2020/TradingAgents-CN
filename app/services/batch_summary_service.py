"""Batch analysis summary construction service."""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from app.core.database import get_mongo_db
from app.services.config_service import ConfigService

logger = logging.getLogger("app.services.batch_summary_service")
config_service = ConfigService()


class BatchSummaryService:
    """Build and persist aggregate summaries for batch analysis tasks."""

    def __init__(self, db_getter: Optional[Callable[[], Any]] = None):
        self._db_getter = db_getter

    def _get_user_id_candidates(self, user_id: str) -> List[Any]:
        """生成用户ID的字符串/ObjectId候选值，兼容历史数据。"""
        candidates: List[Any] = [user_id]
        try:
            from bson import ObjectId
            if str(user_id) == "admin":
                admin_oid_str = "507f1f77bcf86cd799439011"
                candidates.extend([ObjectId(admin_oid_str), admin_oid_str])
            else:
                candidates.append(ObjectId(user_id))
        except Exception:
            pass
        return candidates

    def _normalize_batch_action(self, action: Any, recommendation: str = "") -> Optional[str]:
        """统一批次汇总中的投资建议。"""
        raw = str(action or "").strip()
        text = f"{raw} {recommendation or ''}".lower()
        if any(token in text for token in ["买入", "增持", "buy", "strong buy"]):
            return "买入"
        if any(token in text for token in ["卖出", "减持", "sell", "strong sell"]):
            return "卖出"
        if any(token in text for token in ["持有", "观望", "hold", "neutral"]):
            return "持有"
        return raw or None

    def _safe_batch_number(self, value: Any) -> Optional[float]:
        """将目标价和置信度安全转换为数字。"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            import re
            match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
            if not match:
                return None
            return float(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _normalize_currency_map(data: Any) -> Dict[str, float]:
        if not isinstance(data, dict):
            return {}
        result: Dict[str, float] = {}
        for currency, value in data.items():
            try:
                result[str(currency).upper()] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    def _build_official_batch_cost(
        self,
        status: str,
        before_snapshot: Optional[Dict[str, Any]],
        after_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if status not in ["completed", "partial_success", "failed"]:
            return {
                "available": False,
                "reason": "批次未完成，DeepSeek 余额差额仍在计算中",
                "cost_by_currency": {},
            }
        if not before_snapshot:
            return {"available": False, "reason": "无开始余额快照", "cost_by_currency": {}}
        if not after_snapshot:
            return {"available": False, "reason": "无结束余额快照", "cost_by_currency": {}}
        if not before_snapshot.get("available"):
            return {
                "available": False,
                "reason": before_snapshot.get("error") or "开始余额查询不可用",
                "cost_by_currency": {},
                "before": before_snapshot,
                "after": after_snapshot,
            }
        if not after_snapshot.get("available"):
            return {
                "available": False,
                "reason": after_snapshot.get("error") or "结束余额查询不可用",
                "cost_by_currency": {},
                "before": before_snapshot,
                "after": after_snapshot,
            }

        before = self._normalize_currency_map(before_snapshot.get("balances_by_currency"))
        after = self._normalize_currency_map(after_snapshot.get("balances_by_currency"))
        cost_by_currency: Dict[str, float] = {}
        warnings: List[str] = []
        for currency in sorted(set(before.keys()) | set(after.keys())):
            delta = before.get(currency, 0.0) - after.get(currency, 0.0)
            if delta < -0.000001:
                warnings.append(f"{currency} 余额增加，差额可能受充值或并发调用影响")
                cost_by_currency[currency] = 0.0
            else:
                cost_by_currency[currency] = round(max(delta, 0.0), 6)

        return {
            "available": True,
            "cost_by_currency": cost_by_currency,
            "warning": "；".join(warnings) if warnings else None,
            "before": before_snapshot,
            "after": after_snapshot,
        }

    async def _build_local_batch_cost(
        self,
        db: Any,
        batch_doc: Dict[str, Any],
        task_docs: Dict[str, Dict[str, Any]],
        report_docs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        task_ids = list(task_docs.keys())
        cost_by_currency: Dict[str, float] = {}
        input_tokens = 0
        output_tokens = 0
        usage_records_count = 0

        try:
            if task_ids and hasattr(db, "token_usage"):
                cursor = db.token_usage.find({"session_id": {"$in": task_ids}})
                async for record in cursor:
                    currency = str(record.get("currency") or "CNY").upper()
                    cost_by_currency[currency] = cost_by_currency.get(currency, 0.0) + float(record.get("cost") or 0.0)
                    input_tokens += int(record.get("input_tokens") or 0)
                    output_tokens += int(record.get("output_tokens") or 0)
                    usage_records_count += 1
        except Exception as e:
            logger.warning("读取批次 token_usage 失败: %s", e)

        if usage_records_count:
            return {
                "available": True,
                "source": "token_usage",
                "cost_by_currency": {k: round(v, 6) for k, v in cost_by_currency.items()},
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        tokens_used_total = 0
        for task_id, task in task_docs.items():
            report = report_docs.get(task_id) or {}
            report_tokens = int(report.get("tokens_used") or 0)
            task_result = task.get("result") or {}
            tokens_used_total += report_tokens or int(task_result.get("tokens_used") or task.get("tokens_used") or 0)

        if tokens_used_total <= 0:
            return {"available": False, "reason": "没有可用于估算的 token 使用记录", "cost_by_currency": {}}

        input_tokens = tokens_used_total // 2
        output_tokens = tokens_used_total - input_tokens
        params = batch_doc.get("parameters") or {}
        model_name = params.get("deep_analysis_model") or params.get("quick_analysis_model")
        for report in report_docs.values():
            report_model = report.get("model_info")
            if report_model and report_model != "Unknown":
                model_name = report_model
                break
        if not model_name:
            return {
                "available": False,
                "reason": "无法确定模型价格配置",
                "cost_by_currency": {},
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        try:
            config = await config_service.get_system_config()
            llm_config = None
            for item in config.llm_configs if config else []:
                if item.model_name == model_name:
                    llm_config = item
                    break
            if not llm_config:
                return {
                    "available": False,
                    "reason": f"未找到模型 {model_name} 的价格配置",
                    "cost_by_currency": {},
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

            input_price = llm_config.input_price_per_1k or 0.0
            output_price = llm_config.output_price_per_1k or 0.0
            currency = (llm_config.currency or "CNY").upper()
            cost = (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)
            return {
                "available": True,
                "source": "report_tokens",
                "model_name": model_name,
                "cost_by_currency": {currency: round(cost, 6)},
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            return {
                "available": False,
                "reason": f"本地成本估算失败: {e}",
                "cost_by_currency": {},
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

    async def get_batch_summary(self, user_id: str, batch_id: str) -> Optional[Dict[str, Any]]:
        """获取或重算批量分析总体报告。"""
        try:
            db = (self._db_getter or get_mongo_db)()
            uid_candidates = self._get_user_id_candidates(user_id)
            batch_doc = await db.analysis_batches.find_one({
                "batch_id": batch_id,
                "$or": [
                    {"user_id": {"$in": uid_candidates}},
                    {"user": {"$in": uid_candidates}},
                ]
            })
            if not batch_doc:
                return None

            if batch_doc.get("batch_type") == "quick_decision" and batch_doc.get("results_summary"):
                return batch_doc["results_summary"]

            mapping = batch_doc.get("mapping") or []
            task_ids = [item.get("task_id") for item in mapping if item.get("task_id")]
            if not task_ids:
                task_ids = batch_doc.get("task_ids") or []

            task_docs: Dict[str, Dict[str, Any]] = {}
            if task_ids:
                cursor = db.analysis_tasks.find({"task_id": {"$in": task_ids}})
            else:
                cursor = db.analysis_tasks.find({"batch_id": batch_id})
            async for doc in cursor:
                task_docs[doc.get("task_id")] = doc

            report_docs: Dict[str, Dict[str, Any]] = {}
            if task_docs:
                cursor = db.analysis_reports.find({"task_id": {"$in": list(task_docs.keys())}})
                async for doc in cursor:
                    report_docs[doc.get("task_id")] = doc

            ordered_items = mapping or [
                {
                    "task_id": task_id,
                    "symbol": task.get("symbol") or task.get("stock_code") or task.get("stock_symbol"),
                    "stock_code": task.get("symbol") or task.get("stock_code") or task.get("stock_symbol"),
                }
                for task_id, task in task_docs.items()
            ]

            rows: List[Dict[str, Any]] = []
            completed_count = 0
            failed_count = 0

            for item in ordered_items:
                task_id = item.get("task_id")
                task = task_docs.get(task_id, {}) if task_id else {}
                report = report_docs.get(task_id, {}) if task_id else {}
                result = task.get("result") or {}
                decision = report.get("decision") or result.get("decision") or {}
                recommendation_text = report.get("recommendation") or result.get("recommendation") or ""
                raw_status = task.get("status") or report.get("status") or "pending"
                status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
                symbol = (
                    item.get("symbol") or item.get("stock_code") or
                    task.get("symbol") or task.get("stock_code") or task.get("stock_symbol") or
                    report.get("stock_symbol")
                )

                is_completed = status == "completed"
                is_failed = status == "failed"
                if is_completed:
                    completed_count += 1
                if is_failed:
                    failed_count += 1

                report_object_id = report.get("_id")
                report_id = str(report_object_id) if report_object_id else (
                    report.get("analysis_id") or result.get("analysis_id") or task_id
                )

                rows.append({
                    "task_id": task_id,
                    "report_id": report_id if is_completed and report_id else None,
                    "analysis_id": report.get("analysis_id") or result.get("analysis_id"),
                    "report_available": bool(is_completed and report_id),
                    "symbol": symbol,
                    "stock_code": symbol,
                    "stock_name": task.get("stock_name") or report.get("stock_name") or "",
                    "status": status,
                    "recommendation": (
                        self._normalize_batch_action(decision.get("action") if isinstance(decision, dict) else None, recommendation_text)
                        if is_completed else None
                    ),
                    "target_price": (
                        self._safe_batch_number(decision.get("target_price") if isinstance(decision, dict) else None)
                        if is_completed else None
                    ),
                    "confidence": (
                        self._safe_batch_number(
                            (decision.get("confidence") if isinstance(decision, dict) else None)
                            if (isinstance(decision, dict) and decision.get("confidence") is not None)
                            else (report.get("confidence_score") if report else result.get("confidence_score"))
                        )
                        if is_completed else None
                    ),
                    "error_message": (
                        task.get("last_error") or task.get("error_message") or task.get("message")
                    ) if is_failed else None,
                })

            total_tasks = int(batch_doc.get("total_tasks") or len(rows))
            pending_count = max(total_tasks - completed_count - failed_count, 0)
            if total_tasks == 0:
                status = "failed"
            elif failed_count == total_tasks:
                status = "failed"
            elif completed_count + failed_count >= total_tasks:
                status = "partial_success" if failed_count else "completed"
            else:
                status = "processing"

            if status in ["completed", "partial_success", "failed"] and not batch_doc.get("deepseek_balance_after"):
                try:
                    from app.services.deepseek_balance_service import deepseek_balance_service
                    after_snapshot = await deepseek_balance_service.get_balance_snapshot()
                    batch_doc["deepseek_balance_after"] = after_snapshot
                except Exception as e:
                    logger.warning("懒加载 DeepSeek 结束余额失败: %s", e)

            official_cost = self._build_official_batch_cost(
                status,
                batch_doc.get("deepseek_balance_before"),
                batch_doc.get("deepseek_balance_after"),
            )
            local_cost = await self._build_local_batch_cost(db, batch_doc, task_docs, report_docs)
            cost_summary = {
                "official_available": bool(official_cost.get("available")),
                "official_cost_delta_by_currency": official_cost.get("cost_by_currency") or {},
                "official_reason": official_cost.get("reason"),
                "balance_warning": official_cost.get("warning"),
                "local_estimate_available": bool(local_cost.get("available")),
                "local_estimated_cost_by_currency": local_cost.get("cost_by_currency") or {},
                "local_estimate_reason": local_cost.get("reason"),
                "local_estimate_source": local_cost.get("source"),
                "input_tokens": local_cost.get("input_tokens"),
                "output_tokens": local_cost.get("output_tokens"),
                "deepseek_balance_before": batch_doc.get("deepseek_balance_before"),
                "deepseek_balance_after": batch_doc.get("deepseek_balance_after"),
            }

            generated_at = datetime.utcnow()
            summary = {
                "batch_id": batch_id,
                "title": batch_doc.get("title") or "批量分析",
                "description": batch_doc.get("description"),
                "status": status,
                "total_tasks": total_tasks,
                "completed_tasks": completed_count,
                "failed_tasks": failed_count,
                "pending_tasks": pending_count,
                "generated_at": generated_at.isoformat(),
                "cost_summary": cost_summary,
                "official_cost_delta_by_currency": cost_summary["official_cost_delta_by_currency"],
                "local_estimated_cost_by_currency": cost_summary["local_estimated_cost_by_currency"],
                "items": rows,
            }

            update_data = {
                "status": status,
                "completed_tasks": completed_count,
                "failed_tasks": failed_count,
                "progress": int(((completed_count + failed_count) / total_tasks) * 100) if total_tasks else 0,
                "results_summary": summary,
                "cost_summary": cost_summary,
                "official_cost_delta_by_currency": cost_summary["official_cost_delta_by_currency"],
                "local_estimated_cost_by_currency": cost_summary["local_estimated_cost_by_currency"],
                "updated_at": generated_at,
                **({"completed_at": generated_at} if status in ["completed", "partial_success", "failed"] else {})
            }
            if batch_doc.get("deepseek_balance_after"):
                update_data["deepseek_balance_after"] = batch_doc.get("deepseek_balance_after")

            await db.analysis_batches.update_one(
                {"batch_id": batch_id},
                {"$set": update_data}
            )

            return summary
        except Exception as e:
            logger.error(f"❌ 获取批次汇总失败: {batch_id} - {e}", exc_info=True)
            return None
