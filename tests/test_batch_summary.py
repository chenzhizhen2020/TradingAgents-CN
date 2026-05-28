import pytest

from app.services.batch_summary_service import BatchSummaryService


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
    def __init__(self, docs):
        self.docs = docs
        self.last_update = None

    async def find_one(self, query, projection=None):
        batch_id = query.get("batch_id")
        task_id = query.get("task_id")
        for doc in self.docs:
            if batch_id and doc.get("batch_id") != batch_id:
                continue
            if task_id and doc.get("task_id") != task_id:
                continue
            return doc
        return None

    def find(self, query):
        docs = self.docs
        if "task_id" in query and "$in" in query["task_id"]:
            allowed = set(query["task_id"]["$in"])
            docs = [doc for doc in docs if doc.get("task_id") in allowed]
        elif query.get("batch_id"):
            docs = [doc for doc in docs if doc.get("batch_id") == query["batch_id"]]
        return _AsyncCursor(docs)

    async def update_one(self, query, update):
        self.last_update = {"query": query, "update": update}
        return None


class _DB:
    def __init__(self, batches, tasks, reports, token_usage=None):
        self.analysis_batches = _Collection(batches)
        self.analysis_tasks = _Collection(tasks)
        self.analysis_reports = _Collection(reports)
        if token_usage is not None:
            self.token_usage = _Collection(token_usage)


def _batch_summary_service(db):
    return BatchSummaryService(db_getter=lambda: db)


@pytest.mark.asyncio
async def test_batch_summary_service_returns_summary_contract():
    db = _DB(
        batches=[{
            "batch_id": "B0",
            "user_id": "u1",
            "title": "抽取服务测试",
            "total_tasks": 1,
            "mapping": [{"task_id": "T1", "symbol": "000001"}],
        }],
        tasks=[{
            "task_id": "T1",
            "batch_id": "B0",
            "stock_code": "000001",
            "stock_name": "平安银行",
            "status": "completed",
            "result": {"decision": {"action": "BUY", "target_price": "12.34", "confidence": 0.9}},
        }],
        reports=[],
    )

    summary = await _batch_summary_service(db).get_batch_summary("u1", "B0")

    assert summary["batch_id"] == "B0"
    assert summary["status"] == "completed"
    assert summary["items"][0]["recommendation"] == "买入"
    assert summary["items"][0]["target_price"] == 12.34
    assert db.analysis_batches.last_update["update"]["$set"]["results_summary"]["batch_id"] == "B0"


@pytest.mark.asyncio
async def test_simple_analysis_service_keeps_batch_summary_wrapper(monkeypatch):
    from app.services.simple_analysis_service import SimpleAnalysisService

    db = _DB(
        batches=[{
            "batch_id": "B-wrapper",
            "user_id": "u1",
            "title": "兼容包装",
            "total_tasks": 1,
            "mapping": [{"task_id": "T1", "symbol": "000001"}],
        }],
        tasks=[{
            "task_id": "T1",
            "batch_id": "B-wrapper",
            "stock_code": "000001",
            "status": "completed",
            "result": {"decision": {"action": "BUY"}},
        }],
        reports=[],
    )
    monkeypatch.setattr("app.services.simple_analysis_service.get_mongo_db", lambda: db)
    service = SimpleAnalysisService.__new__(SimpleAnalysisService)

    summary = await service.get_batch_summary("u1", "B-wrapper")

    assert summary["batch_id"] == "B-wrapper"
    assert summary["status"] == "completed"


@pytest.mark.asyncio
async def test_batch_summary_counts_partial_success_and_extracts_fields():
    db = _DB(
        batches=[{
            "batch_id": "B1",
            "user_id": "u1",
            "title": "测试批次",
            "total_tasks": 2,
            "mapping": [
                {"task_id": "T1", "symbol": "000001"},
                {"task_id": "T2", "symbol": "000002"},
            ],
        }],
        tasks=[
            {
                "task_id": "T1",
                "batch_id": "B1",
                "stock_code": "000001",
                "stock_name": "平安银行",
                "status": "completed",
                "result": {
                    "decision": {"action": "BUY", "target_price": "12.34元", "confidence": 0.82},
                    "recommendation": "投资建议：买入。",
                },
            },
            {
                "task_id": "T2",
                "batch_id": "B1",
                "stock_code": "000002",
                "stock_name": "万科A",
                "status": "failed",
                "last_error": "数据源失败",
            },
        ],
        reports=[],
    )

    summary = await _batch_summary_service(db).get_batch_summary("u1", "B1")

    assert summary["status"] == "partial_success"
    assert summary["completed_tasks"] == 1
    assert summary["failed_tasks"] == 1
    assert summary["items"][0]["recommendation"] == "买入"
    assert summary["items"][0]["target_price"] == 12.34
    assert summary["items"][0]["confidence"] == 0.82
    assert summary["items"][1]["error_message"] == "数据源失败"
    assert db.analysis_batches.last_update["update"]["$set"]["results_summary"]["batch_id"] == "B1"


@pytest.mark.asyncio
async def test_batch_summary_uses_report_decision_before_task_result():
    db = _DB(
        batches=[{
            "batch_id": "B2",
            "user_id": "u1",
            "title": "测试批次",
            "total_tasks": 1,
            "mapping": [{"task_id": "T1", "symbol": "600000"}],
        }],
        tasks=[{
            "task_id": "T1",
            "batch_id": "B2",
            "stock_code": "600000",
            "status": "completed",
            "result": {
                "decision": {"action": "SELL", "target_price": 8, "confidence": 0.4},
                "confidence_score": 0.4,
            },
        }],
        reports=[{
            "task_id": "T1",
            "decision": {"action": "HOLD", "target_price": "10.50", "confidence": None},
            "confidence_score": 0.66,
        }],
    )

    summary = await _batch_summary_service(db).get_batch_summary("u1", "B2")

    item = summary["items"][0]
    assert summary["status"] == "completed"
    assert item["recommendation"] == "持有"
    assert item["target_price"] == 10.5
    assert item["confidence"] == 0.66


@pytest.mark.asyncio
async def test_batch_summary_includes_report_link_and_balance_delta():
    db = _DB(
        batches=[{
            "batch_id": "B3",
            "user_id": "u1",
            "title": "费用批次",
            "total_tasks": 1,
            "mapping": [{"task_id": "T1", "symbol": "000001"}],
            "deepseek_balance_before": {
                "available": True,
                "balances_by_currency": {"CNY": 10.0},
            },
            "deepseek_balance_after": {
                "available": True,
                "balances_by_currency": {"CNY": 9.25},
            },
        }],
        tasks=[{
            "task_id": "T1",
            "batch_id": "B3",
            "stock_code": "000001",
            "stock_name": "平安银行",
            "status": "completed",
            "result": {"decision": {"action": "BUY"}},
        }],
        reports=[{
            "_id": "report-object-id",
            "task_id": "T1",
            "analysis_id": "A1",
            "decision": {"action": "BUY"},
        }],
    )

    summary = await _batch_summary_service(db).get_batch_summary("u1", "B3")

    item = summary["items"][0]
    assert item["report_available"] is True
    assert item["report_id"] == "report-object-id"
    assert item["analysis_id"] == "A1"
    assert summary["cost_summary"]["official_available"] is True
    assert summary["cost_summary"]["official_cost_delta_by_currency"] == {"CNY": 0.75}


@pytest.mark.asyncio
async def test_batch_summary_warns_when_balance_increases():
    db = _DB(
        batches=[{
            "batch_id": "B4",
            "user_id": "u1",
            "title": "余额异常批次",
            "total_tasks": 1,
            "mapping": [{"task_id": "T1", "symbol": "000001"}],
            "deepseek_balance_before": {
                "available": True,
                "balances_by_currency": {"CNY": 8.0},
            },
            "deepseek_balance_after": {
                "available": True,
                "balances_by_currency": {"CNY": 9.0},
            },
        }],
        tasks=[{
            "task_id": "T1",
            "batch_id": "B4",
            "stock_code": "000001",
            "status": "completed",
            "result": {"decision": {"action": "HOLD"}},
        }],
        reports=[],
    )

    summary = await _batch_summary_service(db).get_batch_summary("u1", "B4")

    assert summary["cost_summary"]["official_available"] is True
    assert summary["cost_summary"]["official_cost_delta_by_currency"] == {"CNY": 0.0}
    assert "余额增加" in summary["cost_summary"]["balance_warning"]
