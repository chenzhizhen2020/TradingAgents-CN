"""Fast batch buy/sell/hold decision service."""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.core.database import get_mongo_db
from app.models.analysis import BatchAnalysisRequest

logger = logging.getLogger("app.services.quick_batch_decision_service")

LLMDecider = Callable[[List[Dict[str, Any]], str, Dict[str, Any]], Awaitable[Any]]
AgentLLM = Callable[
    [str, str, List[Dict[str, Any]], List[Dict[str, str]], str, Dict[str, Any], bool],
    Awaitable[Any],
]


class QuickBatchDecisionService:
    """Build a fast batch decision from batch quote rows and multi-agent discussion."""

    max_symbols = 10

    def __init__(
        self,
        db_getter: Optional[Callable[[], Any]] = None,
        llm_decider: Optional[LLMDecider] = None,
        agent_llm: Optional[AgentLLM] = None,
        quote_fetcher: Optional[Callable[[List[str]], Awaitable[Dict[str, Dict[str, Any]]]]] = None,
    ):
        self._db_getter = db_getter
        self._llm_decider = llm_decider
        self._agent_llm = agent_llm
        self._quote_fetcher = quote_fetcher

    async def run(self, user_id: str, request: BatchAnalysisRequest) -> Dict[str, Any]:
        started = time.perf_counter()
        symbols = self._normalize_symbols(request.get_symbols())
        if not symbols:
            raise ValueError("股票代码列表不能为空")
        if len(symbols) > self.max_symbols:
            raise ValueError(f"快速批量决策最多支持 {self.max_symbols} 只股票")

        db = (self._db_getter or get_mongo_db)()
        quick_model_name, deep_model_name = self._resolve_model_names(request)
        parameters = request.parameters.model_dump() if request.parameters else {}
        batch_id = str(uuid.uuid4())

        data_by_symbol = await self._load_stock_data(db, symbols)
        missing_symbols = [symbol for symbol in symbols if symbol not in data_by_symbol]
        if missing_symbols:
            fallback_data = await self._fetch_missing_quotes(missing_symbols)
            data_by_symbol.update(fallback_data)

        available_rows = [data_by_symbol[symbol] for symbol in symbols if symbol in data_by_symbol]
        decision_map: Dict[str, Dict[str, Any]] = {}
        discussion_trace: List[Dict[str, str]] = []
        if available_rows:
            raw_decision, discussion_trace = await self._decide_with_agents(
                available_rows,
                quick_model_name,
                deep_model_name,
                parameters,
            )
            decision_map = self._parse_llm_decisions(raw_decision)

        items: List[Dict[str, Any]] = []
        for symbol in symbols:
            data = data_by_symbol.get(symbol)
            if not data:
                items.append(self._missing_item(symbol))
                continue

            decision = decision_map.get(symbol)
            if not decision:
                raise ValueError(f"快速批量决策模型未返回 {symbol} 的决策")
            items.append(self._build_item(symbol, data, decision))

        status = "partial_success" if any(item["data_status"] != "ok" for item in items) else "completed"
        generated_at = datetime.utcnow()
        result = {
            "batch_id": batch_id,
            "batch_type": "quick_decision",
            "title": request.title,
            "description": request.description,
            "status": status,
            "total_tasks": len(symbols),
            "completed_tasks": sum(1 for item in items if item["data_status"] == "ok"),
            "failed_tasks": sum(1 for item in items if item["data_status"] != "ok"),
            "pending_tasks": 0,
            "generated_at": generated_at.isoformat(),
            "items": items,
            "summary": {
                "action_counts": self._count_actions(items),
                "model_name": quick_model_name,
                "deep_model_name": deep_model_name,
                "discussion_rounds": len(discussion_trace),
                "discussion_agents": [entry["agent"] for entry in discussion_trace],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "generated_at": generated_at.isoformat(),
            },
            "discussion_trace": discussion_trace,
        }

        await db.analysis_batches.insert_one({
            "batch_id": batch_id,
            "batch_type": "quick_decision",
            "user_id": user_id,
            "title": request.title,
            "description": request.description,
            "status": status,
            "total_tasks": len(symbols),
            "completed_tasks": result["completed_tasks"],
            "failed_tasks": result["failed_tasks"],
            "cancelled_tasks": 0,
            "progress": 100,
            "symbols": symbols,
            "parameters": parameters,
            "results_summary": result,
            "created_at": generated_at,
            "started_at": generated_at,
            "completed_at": generated_at,
            "updated_at": generated_at,
        })
        return result

    def _normalize_symbols(self, raw_symbols: List[str]) -> List[str]:
        symbols: List[str] = []
        seen = set()
        for raw in raw_symbols:
            digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
            symbol = digits[-6:] if len(digits) >= 6 else digits.zfill(6) if digits else ""
            if not re.fullmatch(r"\d{6}", symbol):
                raise ValueError(f"仅支持 A 股 6 位股票代码: {raw}")
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
        return symbols

    def _resolve_model_names(self, request: BatchAnalysisRequest) -> tuple[str, str]:
        params = request.parameters
        quick_model_name = getattr(params, "quick_analysis_model", None) if params else None
        deep_model_name = getattr(params, "deep_analysis_model", None) if params else None
        quick_model_name = quick_model_name or "qwen-turbo"
        return quick_model_name, deep_model_name or quick_model_name

    def _resolve_model_name(self, request: BatchAnalysisRequest) -> str:
        quick_model_name, _ = self._resolve_model_names(request)
        return quick_model_name

    async def _load_stock_data(self, db: Any, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        data: Dict[str, Dict[str, Any]] = {}
        for collection_name in ("stock_screening_view", "market_quotes"):
            try:
                collection = db[collection_name]
                cursor = collection.find({"code": {"$in": symbols}})
                async for doc in cursor:
                    symbol = str(doc.get("code") or doc.get("symbol") or "")
                    if symbol in symbols and symbol not in data:
                        data[symbol] = self._normalize_data_row(symbol, doc)
            except Exception as e:
                logger.warning("读取 %s 失败: %s", collection_name, e)
        return data

    async def _fetch_missing_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        if self._quote_fetcher:
            rows = await self._quote_fetcher(symbols)
        else:
            try:
                from tradingagents.dataflows.providers.china.akshare import get_akshare_provider

                rows = await get_akshare_provider().get_batch_stock_quotes(symbols)
            except Exception as e:
                logger.warning("AkShare 批量行情兜底失败: %s", e)
                rows = {}
        return {
            symbol: self._normalize_data_row(symbol, row)
            for symbol, row in (rows or {}).items()
            if symbol in symbols and row
        }

    def _normalize_data_row(self, symbol: str, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "stock_name": row.get("name") or row.get("stock_name") or f"股票{symbol}",
            "close": self._safe_float(row.get("close", row.get("price"))),
            "pct_chg": self._safe_float(row.get("pct_chg", row.get("change_percent"))),
            "amount": self._safe_float(row.get("amount")),
            "turnover_rate": self._safe_float(row.get("turnover_rate")),
            "volume_ratio": self._safe_float(row.get("volume_ratio")),
            "pe": self._safe_float(row.get("pe")),
            "pe_ttm": self._safe_float(row.get("pe_ttm")),
            "pb": self._safe_float(row.get("pb")),
            "total_mv": self._safe_float(row.get("total_mv")),
            "circ_mv": self._safe_float(row.get("circ_mv")),
            "data_source": row.get("source") or row.get("data_source") or "local",
        }

    async def _decide_with_agents(
        self,
        rows: List[Dict[str, Any]],
        quick_model_name: str,
        deep_model_name: str,
        parameters: Dict[str, Any],
    ) -> tuple[Any, List[Dict[str, str]]]:
        if self._llm_decider:
            raw_decision = await self._llm_decider(rows, quick_model_name, parameters)
            return raw_decision, [{
                "agent": "组合决策员",
                "role": "portfolio_manager",
                "content": "使用兼容决策器直接生成结果",
            }]

        discussion: List[Dict[str, str]] = []
        roles = self._discussion_roles(parameters)
        raw_decision: Any = None

        for index, role in enumerate(roles):
            is_final = index == len(roles) - 1
            response = await self._call_agent(
                role["key"],
                role["name"],
                rows,
                discussion,
                deep_model_name if role.get("model") == "deep" else quick_model_name,
                parameters,
                final=is_final,
            )
            content = self._agent_content(response)
            discussion.append({
                "agent": role["name"],
                "role": role["key"],
                "content": content,
            })
            if is_final:
                raw_decision = response

        return raw_decision, discussion

    def _discussion_roles(self, parameters: Dict[str, Any]) -> List[Dict[str, str]]:
        analyst_map = {
            "market": {
                "key": "market_analyst",
                "name": "市场分析师",
                "instruction": "从价格、涨跌幅、成交额、换手率、量比等维度横向比较整批股票，指出相对强弱和异常信号。",
            },
            "fundamentals": {
                "key": "fundamentals_analyst",
                "name": "基本面分析师",
                "instruction": "从 PE、PE_TTM、PB、市值等估值指标横向比较整批股票，指出估值优势、估值陷阱和缺失信息。",
            },
            "news": {
                "key": "news_analyst",
                "name": "新闻分析师",
                "instruction": "在已有行情/估值数据有限的前提下，指出整批股票需要关注的事件催化、政策和行业新闻风险；不得编造具体新闻。",
            },
            "social": {
                "key": "social_analyst",
                "name": "情绪分析师",
                "instruction": "从市场情绪、资金偏好和短期交易热度角度横向比较整批股票；数据不足时明确说明不确定性。",
            },
        }
        selected = parameters.get("selected_analysts") or ["market", "fundamentals", "news", "social"]
        analyst_roles = [
            {**analyst_map[key], "model": "quick"}
            for key in selected
            if key in analyst_map
        ]
        if not analyst_roles:
            analyst_roles = [
                {**analyst_map["market"], "model": "quick"},
                {**analyst_map["fundamentals"], "model": "quick"},
            ]

        max_debate_rounds, max_risk_discuss_rounds = self._discussion_round_settings(parameters)

        debate_roles: List[Dict[str, str]] = []
        for _ in range(max_debate_rounds):
            debate_roles.extend([
                {
                    "key": "bull_researcher",
                    "name": "看多研究员",
                    "instruction": "基于前面分析，为整批股票中最值得买入或持有的标的提出看多论据，并比较排序。",
                    "model": "quick",
                },
                {
                    "key": "bear_researcher",
                    "name": "看空研究员",
                    "instruction": "基于前面分析，为整批股票中应卖出或回避的标的提出看空论据，并指出主要风险。",
                    "model": "quick",
                },
            ])

        middle_roles = [
            {
                "key": "research_manager",
                "name": "研究经理",
                "instruction": "综合分析师报告和多空辩论，为整批股票形成研究共识、排序和关键分歧。",
                "model": "deep",
            },
            {
                "key": "trader",
                "name": "交易员",
                "instruction": "把研究共识转成整批股票的交易计划，说明仓位优先级、买卖触发点和不交易条件。",
                "model": "quick",
            },
        ]

        risk_roles: List[Dict[str, str]] = []
        for _ in range(max_risk_discuss_rounds):
            risk_roles.extend([
                {
                    "key": "risky_analyst",
                    "name": "激进风险评估",
                    "instruction": "从激进收益角度审视交易员计划，指出哪些股票值得承担风险，以及收益风险比依据。",
                    "model": "quick",
                },
                {
                    "key": "safe_analyst",
                    "name": "保守风险评估",
                    "instruction": "从保守防守角度审视交易员计划，指出应卖出、回避或降低置信度的股票。",
                    "model": "quick",
                },
                {
                    "key": "neutral_analyst",
                    "name": "中性风险评估",
                    "instruction": "从中性平衡角度校验激进和保守观点，给出折中风险判断。",
                    "model": "quick",
                },
            ])

        final_role = [{
            "key": "risk_manager",
            "name": "风险经理",
            "instruction": "综合所有讨论，做最终风控裁决，并为每只股票输出 BUY、SELL 或 HOLD。",
            "model": "deep",
        }]

        return analyst_roles + debate_roles + middle_roles + risk_roles + final_role

    def _discussion_round_settings(self, parameters: Dict[str, Any]) -> tuple[int, int]:
        depth = self._normalize_research_depth(parameters.get("research_depth"))
        if depth in {"快速", "基础"}:
            return 1, 1
        if depth == "深度":
            return 2, 2
        if depth == "全面":
            return 3, 3
        return 1, 2

    def _normalize_research_depth(self, raw_depth: Any) -> str:
        numeric_to_chinese = {
            1: "快速",
            2: "基础",
            3: "标准",
            4: "深度",
            5: "全面",
        }
        if isinstance(raw_depth, (int, float)):
            return numeric_to_chinese.get(int(raw_depth), "标准")
        if isinstance(raw_depth, str):
            depth = raw_depth.strip()
            if depth.isdigit():
                return numeric_to_chinese.get(int(depth), "标准")
            if depth in {"快速", "基础", "标准", "深度", "全面"}:
                return depth
        return "标准"

    async def _call_agent(
        self,
        role_key: str,
        role_name: str,
        rows: List[Dict[str, Any]],
        discussion: List[Dict[str, str]],
        model_name: str,
        parameters: Dict[str, Any],
        final: bool = False,
    ) -> Any:
        if self._agent_llm:
            return await self._agent_llm(role_key, role_name, rows, discussion, model_name, parameters, final)
        return await self._call_openai_compatible_agent(
            role_key,
            role_name,
            rows,
            discussion,
            model_name,
            parameters,
            final=final,
        )

    async def _decide_with_llm(self, rows: List[Dict[str, Any]], model_name: str, parameters: Dict[str, Any]) -> Any:
        if self._llm_decider:
            return await self._llm_decider(rows, model_name, parameters)
        return await self._call_openai_compatible_llm(rows, model_name, parameters)

    async def _call_openai_compatible_agent(
        self,
        role_key: str,
        role_name: str,
        rows: List[Dict[str, Any]],
        discussion: List[Dict[str, str]],
        model_name: str,
        parameters: Dict[str, Any],
        final: bool = False,
    ) -> str:
        llm = self._create_llm(model_name)
        prompt = self._build_agent_prompt(role_key, role_name, rows, discussion, parameters, final=final)
        response = await llm.ainvoke(prompt)
        return getattr(response, "content", response)

    async def _call_openai_compatible_llm(
        self,
        rows: List[Dict[str, Any]],
        model_name: str,
        parameters: Dict[str, Any],
    ) -> str:
        llm = self._create_llm(model_name)
        prompt = self._build_prompt(rows, parameters)
        response = await llm.ainvoke(prompt)
        return getattr(response, "content", response)

    def _create_llm(self, model_name: str) -> Any:
        try:
            from app.services.simple_analysis_service import get_provider_and_url_by_model_sync
            from tradingagents.llm_clients.openai_client import OpenAIClient
        except Exception as e:
            raise ValueError(f"快速批量决策模型初始化失败: {e}")

        provider_info = get_provider_and_url_by_model_sync(model_name)
        return OpenAIClient(
            model=model_name,
            provider=provider_info.get("provider") or "qwen",
            base_url=provider_info.get("backend_url"),
            api_key=provider_info.get("api_key"),
            temperature=0,
            max_tokens=4500,
            timeout=600,
        ).get_llm()

    def _build_agent_prompt(
        self,
        role_key: str,
        role_name: str,
        rows: List[Dict[str, Any]],
        discussion: List[Dict[str, str]],
        parameters: Dict[str, Any],
        final: bool = False,
    ) -> str:
        language = parameters.get("language") or "zh-CN"
        role = next((item for item in self._discussion_roles(parameters) if item["key"] == role_key), None)
        instruction = role["instruction"] if role else "横向比较整批股票并给出观点。"
        discussion_text = "\n\n".join(
            f"{entry['agent']}:\n{entry['content']}" for entry in discussion
        ) or "暂无，当前是第一位发言者。"

        if final:
            output_rule = (
                "你是最后的组合决策员。只输出严格 JSON，不要 Markdown，不要额外解释。\n"
                "JSON 格式: {\"items\":[{\"symbol\":\"000001\",\"action\":\"BUY|SELL|HOLD\","
                "\"confidence\":0.0到1.0,\"target_price\":数字,"
                "\"price_prediction\":{\"current_price\":数字,\"target_price\":数字,"
                "\"stop_loss_price\":数字,\"take_profit_price\":数字,\"watch_price\":数字,"
                "\"expected_change_percent\":数字,\"time_horizon\":\"2-6周|1-3个月|3-6个月\","
                "\"basis\":\"价格预测依据\"},"
                "\"reasoning\":\"结合多角色讨论的一句话原因\"}]}\n"
                "必须覆盖股票数据中每一只 symbol。每只股票必须提供 price_prediction，"
                "所有价格字段必须是数字且不能为 null。BUY 的 target_price 表示上涨目标价；"
                "SELL 的 target_price 表示下行/回落目标价，watch_price 表示重新观察或止跌确认价；"
                "HOLD 的 target_price 表示合理价值中枢，watch_price 表示突破或跌破观察价。"
            )
        else:
            output_rule = (
                "输出一段结构化中文讨论，必须横向比较股票数据中的所有 symbol，"
                "不要输出最终 JSON，不要只分析单只股票。"
            )

        return (
            f"你是{role_name}，正在参与一个批量股票多 Agent 讨论。\n"
            f"语言: {language}\n"
            f"职责: {instruction}\n"
            "讨论对象: 下方股票数据中的全部 A 股，必须作为一个组合一起比较，不允许逐只孤立给结论。\n"
            f"股票数据: {json.dumps(rows, ensure_ascii=False, default=str)}\n\n"
            f"已有讨论:\n{discussion_text}\n\n"
            f"{output_rule}"
        )

    def _build_prompt(self, rows: List[Dict[str, Any]], parameters: Dict[str, Any]) -> str:
        language = parameters.get("language") or "zh-CN"
        return (
            "你是股票组合快速决策助手。根据下方每只 A 股的行情和估值指标，"
            "为每只股票给出 BUY、SELL 或 HOLD。只输出严格 JSON，不要 Markdown。\n"
            f"语言: {language}\n"
            "JSON 格式: {\"items\":[{\"symbol\":\"000001\",\"action\":\"BUY|SELL|HOLD\","
            "\"confidence\":0.0到1.0,\"target_price\":数字,"
            "\"price_prediction\":{\"current_price\":数字,\"target_price\":数字,"
            "\"stop_loss_price\":数字,\"take_profit_price\":数字,\"watch_price\":数字,"
            "\"expected_change_percent\":数字,\"time_horizon\":\"2-6周|1-3个月|3-6个月\","
            "\"basis\":\"价格预测依据\"},\"reasoning\":\"一句话原因\"}]}\n"
            "每只股票必须提供 price_prediction，所有价格字段必须是数字且不能为 null。\n"
            f"股票数据: {json.dumps(rows, ensure_ascii=False, default=str)}"
        )

    def _agent_content(self, response: Any) -> str:
        if isinstance(response, str):
            return response.strip()
        try:
            return json.dumps(response, ensure_ascii=False, default=str)
        except TypeError:
            return str(response)

    def _parse_llm_decisions(self, raw_decision: Any) -> Dict[str, Dict[str, Any]]:
        payload = raw_decision
        if isinstance(raw_decision, str):
            text = raw_decision.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?", "", text).strip()
                text = re.sub(r"```$", "", text).strip()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"快速批量决策模型返回格式无效: {e}")

        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("快速批量决策模型返回格式无效: 缺少 items")

        decisions: Dict[str, Dict[str, Any]] = {}
        for item in payload["items"]:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "")
            if not re.fullmatch(r"\d{6}", symbol):
                continue
            action = self._normalize_action(item.get("action"))
            decisions[symbol] = {
                "action": action,
                "confidence": self._bounded_float(item.get("confidence"), default=0.5),
                "target_price": self._safe_optional_float(item.get("target_price")),
                "price_prediction": item.get("price_prediction"),
                "reasoning": str(item.get("reasoning") or "模型未提供原因").strip(),
            }
        return decisions

    def _build_item(self, symbol: str, data: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        price_prediction = self._normalize_price_prediction(symbol, data, decision)
        return {
            "symbol": symbol,
            "stock_code": symbol,
            "stock_name": data.get("stock_name") or f"股票{symbol}",
            "status": "completed",
            "action": decision["action"],
            "action_label": self._action_label(decision["action"]),
            "confidence": decision["confidence"],
            "target_price": price_prediction["target_price"],
            "price_prediction": price_prediction,
            "reasoning": decision["reasoning"],
            "key_metrics": {
                key: data.get(key)
                for key in ("close", "pct_chg", "pe", "pe_ttm", "pb", "turnover_rate", "volume_ratio", "total_mv")
            },
            "data_status": "ok",
            "data_source": data.get("data_source"),
        }

    def _missing_item(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "stock_code": symbol,
            "stock_name": f"股票{symbol}",
            "status": "failed",
            "action": "HOLD",
            "action_label": "持有",
            "confidence": 0.0,
            "target_price": None,
            "price_prediction": None,
            "reasoning": "未获取到行情/估值数据，无法生成可靠快速决策",
            "key_metrics": {},
            "data_status": "data_missing",
            "data_source": None,
        }

    def _normalize_price_prediction(
        self,
        symbol: str,
        data: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = decision.get("price_prediction")
        if not isinstance(raw, dict):
            raw = {}

        current_price = self._safe_optional_float(raw.get("current_price"))
        if current_price is None:
            current_price = self._safe_optional_float(data.get("close"))
        target_price = self._safe_optional_float(raw.get("target_price"))
        if target_price is None:
            target_price = self._safe_optional_float(decision.get("target_price"))

        prediction = {
            "current_price": current_price,
            "target_price": target_price,
            "stop_loss_price": self._safe_optional_float(raw.get("stop_loss_price")),
            "take_profit_price": self._safe_optional_float(raw.get("take_profit_price")),
            "watch_price": self._safe_optional_float(raw.get("watch_price")),
            "expected_change_percent": self._safe_optional_float(raw.get("expected_change_percent")),
            "time_horizon": str(raw.get("time_horizon") or "").strip(),
            "basis": str(raw.get("basis") or "").strip(),
        }

        if prediction["expected_change_percent"] is None and current_price and target_price:
            prediction["expected_change_percent"] = round((target_price - current_price) / current_price * 100, 2)

        missing_fields = [
            key
            for key in (
                "current_price",
                "target_price",
                "stop_loss_price",
                "take_profit_price",
                "watch_price",
                "expected_change_percent",
            )
            if prediction[key] is None
        ]
        if not prediction["time_horizon"]:
            missing_fields.append("time_horizon")
        if not prediction["basis"]:
            missing_fields.append("basis")
        if missing_fields:
            raise ValueError(f"快速批量决策模型未返回 {symbol} 的完整价格预测信息: {', '.join(missing_fields)}")

        return prediction

    def _count_actions(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for item in items:
            action = item.get("action")
            if action in counts:
                counts[action] += 1
        return counts

    def _normalize_action(self, action: Any) -> str:
        raw = str(action or "").strip().upper()
        if raw in {"BUY", "买入", "增持"}:
            return "BUY"
        if raw in {"SELL", "卖出", "减持"}:
            return "SELL"
        if raw in {"HOLD", "持有", "观望", "NEUTRAL"}:
            return "HOLD"
        raise ValueError(f"快速批量决策模型返回未知动作: {action}")

    def _action_label(self, action: str) -> str:
        return {"BUY": "买入", "SELL": "卖出", "HOLD": "持有"}[action]

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            if isinstance(value, str):
                value = (
                    value.replace("¥", "")
                    .replace("￥", "")
                    .replace("$", "")
                    .replace("元", "")
                    .replace("%", "")
                    .strip()
                )
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_optional_float(self, value: Any) -> Optional[float]:
        return self._safe_float(value)

    def _bounded_float(self, value: Any, default: float) -> float:
        numeric = self._safe_float(value)
        if numeric is None:
            return default
        return max(0.0, min(1.0, numeric))
