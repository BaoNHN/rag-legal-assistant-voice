# regression_test_engine.py
# Background worker: run evaluate/retrieval_regression_tests.py's entity-type
# confusable-pair suite from the admin UI (Manage Law → "Kiểm thử hồi quy" tab),
# mirroring the job/progress pattern used by engine/evaluate_engine.py for the
# "Đánh giá hệ thống RAG" tab — button, progress bar, persisted last-N results.
#
# No LLM calls (see evaluate_cases() in retrieval_regression_tests.py), so a
# run finishes in a couple of seconds — the background-job machinery here is
# mostly for UI consistency with the RAG evaluation tab, not because it's slow.

import os
import sys
import json
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

HISTORY_PATH  = os.path.join(BASE_DIR, "regression_results_history.json")
KEEP_LATEST_N = 2

# ── Job registry ─────────────────────────────────────────────────────────────
_jobs: dict = {}
_lock = threading.Lock()


def get_regression_job(job_id: str) -> dict:
    with _lock:
        return _jobs.get(job_id, {})


def _set(job_id: str, **kwargs):
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


def _load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_run(summary: dict):
    """Prepend the new run and keep only the KEEP_LATEST_N most recent."""
    history = _load_history()
    history.insert(0, summary)
    history = history[:KEEP_LATEST_N]
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass


def get_latest_regression_results() -> list:
    """Up to KEEP_LATEST_N most recent runs, newest first — shown on the
    Manage Law page whether or not a run happens this session."""
    return _load_history()


def run_regression_tests(job_id: str):
    from evaluate.retrieval_regression_tests import evaluate_cases, CASES

    _set(job_id, status="running", message="Đang chạy kiểm thử…", progress=0, summary=None)

    try:
        n = len(CASES)

        def progress_cb(i, total, question):
            pct = round((i / total) * 100)
            _set(job_id, message=f"[{i + 1}/{total}] {question[:55]}…", progress=pct)

        results = evaluate_cases(progress_cb=progress_cb)
        passed  = sum(1 for r in results if r["passed"])

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "passed":    passed,
            "total":     n,
            "results":   results,
        }
        _save_run(summary)

        _set(job_id,
             status="done",
             message=f"✅ Kiểm thử hoàn tất! {passed}/{n} câu đạt.",
             progress=100,
             summary=summary)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set(job_id, status="failed", message=f"❌ Lỗi: {e}", summary=None)
