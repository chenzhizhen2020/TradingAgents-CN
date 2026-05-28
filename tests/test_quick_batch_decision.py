import pytest
from fastapi import HTTPException

from app.models.analysis import BatchAnalysisRequest
from app.services.quick_batch_decision_service import QuickBatchDecisionService


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = docs

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _Collection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.inserted = []

    def find(self, query, projection=None):
        codes = set((query.get("code") or {}).get("$in", []))
        return _AsyncCursor([doc for doc in self.docs if doc.get("code") in codes])

    async def insert_one(self, doc):
        self.inserted.append(doc)
        self.docs.append(doc)


class _DB:
    def __init__(self, screening_docs=None, quote_docs=None):
        self.stock_screening_view = _Collection(screening_docs)
        self.market_quotes = _Collection(quote_docs)
        self.analysis_batches = _Collection()

    def __getitem__(self, name):
        return getattr(self, name)


def _request(symbols):
    return BatchAnalysisRequest(
        title="快速决策",
        symbols=symbols,
        parameters={"quick_analysis_model": "qwen-turbo"},
    )


def _deep_request(symbols):
    return BatchAnalysisRequest(
        title="深度批量决策",
        symbols=symbols,
        parameters={
            "research_depth": "深度",
            "selected_analysts": ["market", "fundamentals", "news", "social"],
            "quick_analysis_model": "qwen-turbo",
            "deep_analysis_model": "qwen-max",
        },
    )


def _price_prediction(current, target, stop_loss, take_profit, watch, basis):
    return {
        "current_price": current,
        "target_price": target,
        "stop_loss_price": stop_loss,
        "take_profit_price": take_profit,
        "watch_price": watch,
        "expected_change_percent": round((target - current) / current * 100, 2),
        "time_horizon": "1-3个月",
        "basis": basis,
    }


async def _llm_json(_rows, _model_name, _parameters):
    return """
    {
      "items": [
        {
          "symbol": "000001",
          "action": "BUY",
          "confidence": 0.86,
          "target_price": 13.2,
          "price_prediction": {
            "current_price": 12.3,
            "target_price": 13.2,
            "stop_loss_price": 11.4,
            "take_profit_price": 13.8,
            "watch_price": 12.8,
            "expected_change_percent": 7.32,
            "time_horizon": "1-3个月",
            "basis": "估值与动量较好"
          },
          "reasoning": "估值与动量较好"
        },
        {
          "symbol": "000002",
          "action": "SELL",
          "confidence": 0.72,
          "target_price": 7.1,
          "price_prediction": {
            "current_price": 8.4,
            "target_price": 7.1,
            "stop_loss_price": 8.8,
            "take_profit_price": 7.3,
            "watch_price": 7.8,
            "expected_change_percent": -15.48,
            "time_horizon": "2-6周",
            "basis": "量价走弱"
          },
          "reasoning": "量价走弱"
        },
        {
          "symbol": "600000",
          "action": "HOLD",
          "confidence": 0.65,
          "target_price": 10.1,
          "price_prediction": {
            "current_price": 9.9,
            "target_price": 10.1,
            "stop_loss_price": 9.3,
            "take_profit_price": 10.8,
            "watch_price": 10.3,
            "expected_change_percent": 2.02,
            "time_horizon": "1-3个月",
            "basis": "信号中性"
          },
          "reasoning": "信号中性"
        }
      ]
    }
    """


@pytest.mark.asyncio
async def test_quick_batch_decision_runs_multi_agent_discussion_over_entire_batch():
    db = _DB(
        screening_docs=[
            {"code": "000001", "name": "平安银行", "close": 12.3, "pct_chg": 1.2, "pe": 6.1, "pb": 0.8},
            {"code": "000002", "name": "万科A", "close": 8.4, "pct_chg": -2.4, "pe": 9.2, "pb": 0.6},
        ]
    )
    calls = []

    async def agent_llm(role_key, _role_name, rows, discussion, _model_name, _parameters, final=False):
        calls.append({
            "role_key": role_key,
            "symbols": [row["symbol"] for row in rows],
            "prior_discussion_count": len(discussion),
            "final": final,
        })
        if final:
            return {
                "items": [
                    {
                        "symbol": "000001",
                        "action": "BUY",
                        "confidence": 0.82,
                        "target_price": 13.1,
                        "price_prediction": _price_prediction(12.3, 13.1, 11.4, 13.6, 12.8, "多方和风控后仍具相对优势"),
                        "reasoning": "多方和风控后仍具相对优势",
                    },
                    {
                        "symbol": "000002",
                        "action": "SELL",
                        "confidence": 0.77,
                        "target_price": 7.2,
                        "price_prediction": _price_prediction(8.4, 7.2, 8.9, 7.4, 7.8, "空方和风控均指出下行压力"),
                        "reasoning": "空方和风控均指出下行压力",
                    },
                ]
            }
        return f"{role_key} 已横向比较 000001 与 000002"

    service = QuickBatchDecisionService(db_getter=lambda: db, agent_llm=agent_llm)

    result = await service.run("u1", _request(["000001", "000002"]))

    assert [call["role_key"] for call in calls] == [
        "market_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "social_analyst",
        "bull_researcher",
        "bear_researcher",
        "research_manager",
        "trader",
        "risky_analyst",
        "safe_analyst",
        "neutral_analyst",
        "risky_analyst",
        "safe_analyst",
        "neutral_analyst",
        "risk_manager",
    ]
    assert all(call["symbols"] == ["000001", "000002"] for call in calls)
    assert calls[-1]["prior_discussion_count"] == 14
    assert calls[-1]["final"] is True
    assert result["summary"]["discussion_rounds"] == 15
    assert result["summary"]["discussion_agents"] == [
        "市场分析师",
        "基本面分析师",
        "新闻分析师",
        "情绪分析师",
        "看多研究员",
        "看空研究员",
        "研究经理",
        "交易员",
        "激进风险评估",
        "保守风险评估",
        "中性风险评估",
        "激进风险评估",
        "保守风险评估",
        "中性风险评估",
        "风险经理",
    ]
    assert len(result["discussion_trace"]) == 15
    assert db.analysis_batches.inserted[0]["results_summary"]["discussion_trace"] == result["discussion_trace"]


@pytest.mark.asyncio
async def test_quick_batch_decision_requires_price_prediction_for_each_action():
    db = _DB(
        screening_docs=[
            {"code": "000001", "name": "平安银行", "close": 12.3, "pct_chg": 1.2, "pe": 6.1, "pb": 0.8},
            {"code": "000002", "name": "万科A", "close": 8.4, "pct_chg": -2.4, "pe": 9.2, "pb": 0.6},
            {"code": "600000", "name": "浦发银行", "close": 9.9, "pct_chg": 0.1, "pe": 5.8, "pb": 0.5},
        ]
    )
    service = QuickBatchDecisionService(db_getter=lambda: db, llm_decider=_llm_json)

    result = await service.run("u1", _request(["000001", "000002", "600000"]))

    for item in result["items"]:
        prediction = item["price_prediction"]
        assert item["target_price"] == prediction["target_price"]
        assert prediction["current_price"] == item["key_metrics"]["close"]
        assert isinstance(prediction["target_price"], float)
        assert isinstance(prediction["stop_loss_price"], float)
        assert isinstance(prediction["take_profit_price"], float)
        assert isinstance(prediction["watch_price"], float)
        assert isinstance(prediction["expected_change_percent"], float)
        assert prediction["time_horizon"]
        assert prediction["basis"]
    assert db.analysis_batches.inserted[0]["results_summary"]["items"][1]["price_prediction"]["target_price"] == 7.1


@pytest.mark.asyncio
async def test_quick_batch_decision_uses_single_stock_style_deep_discussion_rounds():
    db = _DB(
        screening_docs=[
            {"code": "000001", "name": "平安银行", "close": 12.3, "pct_chg": 1.2, "pe": 6.1, "pb": 0.8},
            {"code": "000002", "name": "万科A", "close": 8.4, "pct_chg": -2.4, "pe": 9.2, "pb": 0.6},
        ]
    )
    calls = []

    async def agent_llm(role_key, _role_name, rows, discussion, model_name, _parameters, final=False):
        calls.append({
            "role_key": role_key,
            "symbols": [row["symbol"] for row in rows],
            "model_name": model_name,
            "prior_discussion_count": len(discussion),
            "final": final,
        })
        if final:
            return {
                "items": [
                    {
                        "symbol": "000001",
                        "action": "BUY",
                        "confidence": 0.82,
                        "target_price": 13.1,
                        "price_prediction": _price_prediction(12.3, 13.1, 11.4, 13.6, 12.8, "深度讨论后相对更优"),
                        "reasoning": "深度讨论后相对更优",
                    },
                    {
                        "symbol": "000002",
                        "action": "SELL",
                        "confidence": 0.77,
                        "target_price": 7.2,
                        "price_prediction": _price_prediction(8.4, 7.2, 8.9, 7.4, 7.8, "深度讨论后风险更高"),
                        "reasoning": "深度讨论后风险更高",
                    },
                ]
            }
        return f"{role_key} 批量讨论"

    service = QuickBatchDecisionService(db_getter=lambda: db, agent_llm=agent_llm)

    result = await service.run("u1", _deep_request(["000001", "000002"]))

    expected_roles = [
        "market_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "social_analyst",
        "bull_researcher",
        "bear_researcher",
        "bull_researcher",
        "bear_researcher",
        "research_manager",
        "trader",
        "risky_analyst",
        "safe_analyst",
        "neutral_analyst",
        "risky_analyst",
        "safe_analyst",
        "neutral_analyst",
        "risk_manager",
    ]
    assert [call["role_key"] for call in calls] == expected_roles
    assert all(call["symbols"] == ["000001", "000002"] for call in calls)
    assert calls[-1]["final"] is True
    assert calls[-1]["prior_discussion_count"] == len(expected_roles) - 1
    deep_roles = {"research_manager", "risk_manager"}
    assert all(call["model_name"] == "qwen-max" for call in calls if call["role_key"] in deep_roles)
    assert all(call["model_name"] == "qwen-turbo" for call in calls if call["role_key"] not in deep_roles)
    assert result["summary"]["discussion_rounds"] == len(expected_roles)
    assert result["summary"]["discussion_agents"][-1] == "风险经理"


@pytest.mark.asyncio
async def test_quick_batch_decision_returns_actions_and_persists_batch():
    db = _DB(
        screening_docs=[
            {"code": "000001", "name": "平安银行", "close": 12.3, "pct_chg": 1.2, "pe": 6.1, "pb": 0.8},
            {"code": "000002", "name": "万科A", "close": 8.4, "pct_chg": -2.4, "pe": 9.2, "pb": 0.6},
            {"code": "600000", "name": "浦发银行", "close": 9.9, "pct_chg": 0.1, "pe": 5.8, "pb": 0.5},
        ]
    )
    service = QuickBatchDecisionService(db_getter=lambda: db, llm_decider=_llm_json)

    result = await service.run("u1", _request(["000001", "000002", "600000"]))

    assert result["status"] == "completed"
    assert result["summary"]["action_counts"] == {"BUY": 1, "SELL": 1, "HOLD": 1}
    assert [item["action"] for item in result["items"]] == ["BUY", "SELL", "HOLD"]
    assert result["items"][0]["stock_name"] == "平安银行"
    assert result["items"][0]["data_status"] == "ok"
    assert db.analysis_batches.inserted[0]["batch_type"] == "quick_decision"
    assert db.analysis_batches.inserted[0]["results_summary"]["batch_id"] == result["batch_id"]


@pytest.mark.asyncio
async def test_quick_batch_decision_rejects_more_than_ten_symbols():
    service = QuickBatchDecisionService(db_getter=lambda: _DB(), llm_decider=_llm_json)

    with pytest.raises(ValueError, match="最多支持 10 只股票"):
        await service.run("u1", _request([f"00000{i}" for i in range(11)]))


@pytest.mark.asyncio
async def test_quick_batch_decision_marks_missing_data_as_partial_success():
    db = _DB(
        screening_docs=[
            {"code": "000001", "name": "平安银行", "close": 12.3, "pct_chg": 1.2},
        ]
    )
    async def no_fallback(_symbols):
        return {}

    service = QuickBatchDecisionService(
        db_getter=lambda: db,
        llm_decider=_llm_json,
        quote_fetcher=no_fallback,
    )

    result = await service.run("u1", _request(["000001", "000002"]))

    missing = result["items"][1]
    assert result["status"] == "partial_success"
    assert missing["symbol"] == "000002"
    assert missing["data_status"] == "data_missing"
    assert missing["action"] == "HOLD"
    assert "未获取到行情" in missing["reasoning"]


@pytest.mark.asyncio
async def test_quick_batch_decision_rejects_invalid_llm_json():
    db = _DB(screening_docs=[{"code": "000001", "name": "平安银行", "close": 12.3}])

    async def invalid_json(_rows, _model_name, _parameters):
        return "not-json"

    service = QuickBatchDecisionService(db_getter=lambda: db, llm_decider=invalid_json)

    with pytest.raises(ValueError, match="快速批量决策模型返回格式无效"):
        await service.run("u1", _request(["000001"]))


@pytest.mark.asyncio
async def test_quick_batch_decision_route_returns_400_for_too_many_symbols():
    from app.routers.analysis import submit_quick_batch_decision

    with pytest.raises(HTTPException) as exc_info:
        await submit_quick_batch_decision(
            _request([f"{index:06d}" for index in range(11)]),
            user={"id": "u1"},
        )

    assert exc_info.value.status_code == 400
    assert "最多支持 10 只股票" in exc_info.value.detail
