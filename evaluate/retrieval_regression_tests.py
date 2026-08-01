# retrieval_regression_tests.py
# Regression tests for entity-type retrieval/reranking (see "Lỗi 2/Lỗi 4" in
# GỬI BẢO 25 7 2026.docx). Checks retrieve_docs → filter_compatible_docs →
# select_best_doc directly, WITHOUT calling the LLM — fast, deterministic,
# doesn't touch the Groq quota. Run after any change to rag_engine.py's
# retrieval/rerank logic, or after importing new data into chroma_db.
#
# USAGE:
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python evaluate/retrieval_regression_tests.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.rag_engine import retrieve_docs, filter_compatible_docs, select_best_doc, infer_doc_entity

# Each case: (question, expected_entity_type, forbidden_entity_type)
# forbidden_entity_type = the confusable entity the old code (no hard filter)
# was observed pulling in — the regression this guards against.
CASES = [
    (
        "Thủ tục thành lập công ty TNHH một thành viên là gì?",
        "llc_one_member", "llc_multi_member",
    ),
    (
        "Thủ tục thành lập công ty TNHH hai thành viên trở lên là gì?",
        "llc_multi_member", "llc_one_member",
    ),
    (
        "Điều kiện thành lập công ty cổ phần là gì?",
        "joint_stock_company", "private_enterprise",
    ),
    (
        "Chủ doanh nghiệp tư nhân có quyền và nghĩa vụ gì?",
        "private_enterprise", "joint_stock_company",
    ),
]


def evaluate_cases(cases=CASES, progress_cb=None) -> list:
    """Run every (question, expected, forbidden) case and return a list of
    result dicts. Shared by the CLI entrypoint below and
    engine/regression_test_engine.py (admin UI "Kiểm thử hồi quy" tab on
    Manage Law) so both stay in sync with the same case list and pass/fail
    logic — no LLM calls, so this is fast enough to run synchronously.
    progress_cb(i, n, question), if given, is called before each case runs.
    """
    results = []
    for i, (question, expected, forbidden) in enumerate(cases):
        if progress_cb:
            progress_cb(i, len(cases), question)

        docs = retrieve_docs(question, question)
        filtered = filter_compatible_docs(question, docs)
        best = select_best_doc(question, filtered)

        best_entity = infer_doc_entity(best) if best else None
        filtered_entities = {infer_doc_entity(d) for d in filtered}
        forbidden_present = forbidden in filtered_entities

        passed = best is not None and best_entity == expected and not forbidden_present
        results.append({
            "question":          question,
            "expected":          expected,
            "forbidden":         forbidden,
            "got_best":          best_entity,
            "forbidden_present": forbidden_present,
            "passed":            passed,
        })
    return results


def run():
    results = evaluate_cases()
    failures = [r for r in results if not r["passed"]]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['question']}")
        print(f"        expected={r['expected']}  got_best={r['got_best']}  "
              f"forbidden_in_context={r['forbidden_present']}")

    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    if failures:
        print("FAILED:")
        for r in failures:
            print(f"  - {r['question']}")
        sys.exit(1)


if __name__ == "__main__":
    run()
