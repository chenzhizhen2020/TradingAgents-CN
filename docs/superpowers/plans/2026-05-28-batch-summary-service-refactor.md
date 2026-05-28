# Batch Summary Service Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move batch analysis summary construction out of `SimpleAnalysisService` while preserving the existing API response contract.

**Architecture:** Introduce `app/services/batch_summary_service.py` as the owner of batch summary lookup, row normalization, and cost calculation. Keep `SimpleAnalysisService.get_batch_summary()` as a thin compatibility wrapper so callers and routes do not change.

**Tech Stack:** Python 3.10, FastAPI service layer, Motor-style async collection APIs, pytest, pytest-asyncio.

---

### Task 1: Add Direct Service Test

**Files:**
- Modify: `tests/test_batch_summary.py`
- Create later: `app/services/batch_summary_service.py`

- [ ] **Step 1: Write the failing import/API test**

Add a test that imports `BatchSummaryService`, patches its module-level `get_mongo_db`, and verifies `BatchSummaryService().get_batch_summary("u1", "B1")` returns the same shape already asserted through `SimpleAnalysisService`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m pytest tests/test_batch_summary.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'app.services.batch_summary_service'`.

### Task 2: Extract Batch Summary Service

**Files:**
- Create: `app/services/batch_summary_service.py`
- Modify: `app/services/simple_analysis_service.py`

- [ ] **Step 1: Move helpers**

Move these methods from `SimpleAnalysisService` to `BatchSummaryService` without changing logic:

```python
_get_user_id_candidates
_normalize_batch_action
_safe_batch_number
_normalize_currency_map
_build_official_batch_cost
_build_local_batch_cost
get_batch_summary
```

- [ ] **Step 2: Keep compatibility wrapper**

Replace `SimpleAnalysisService.get_batch_summary()` with:

```python
async def get_batch_summary(self, user_id: str, batch_id: str) -> Optional[Dict[str, Any]]:
    from app.services.batch_summary_service import BatchSummaryService
    return await BatchSummaryService().get_batch_summary(user_id, batch_id)
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m pytest tests/test_batch_summary.py -q
```

Expected: `5 passed`.

### Task 3: Verify Frontend Contract Still Types

**Files:**
- No frontend files expected in this slice.

- [ ] **Step 1: Run type check**

Run:

```bash
npm run type-check
```

Expected: exit code `0`.

### Task 4: Commit And Push

**Files:**
- Stage the new service, modified compatibility wrapper, test update, and this plan.

- [ ] **Step 1: Review diff**

Run:

```bash
git diff --stat
git diff -- app/services/simple_analysis_service.py app/services/batch_summary_service.py tests/test_batch_summary.py
```

- [ ] **Step 2: Commit**

Run:

```bash
git add docs/superpowers/plans/2026-05-28-batch-summary-service-refactor.md app/services/batch_summary_service.py app/services/simple_analysis_service.py tests/test_batch_summary.py
git commit -m "refactor: extract batch summary service"
```

- [ ] **Step 3: Push**

Run:

```bash
git push
```
