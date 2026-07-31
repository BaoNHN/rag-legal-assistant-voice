# evaluate_engine.py
# Background worker: evaluate RAG system quality against a dataset.
# Supports quick (auto, demo-30) and full (llm, all) evaluation modes.

import os
import re
import sys
import json
import threading
import time
import pandas as pd
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# There's no UI to browse historical eval runs, only the most recent one
# matters — keep disk clutter down by pruning older eval_results_*.xlsx files
# every time a new one is written.
KEEP_LATEST_EVAL_FILES = 2
LATEST_RESULT_PATH     = os.path.join(BASE_DIR, "eval_results_latest.json")

# ── Job registry ─────────────────────────────────────────────────────────────
_jobs: dict = {}
_lock = threading.Lock()


def get_eval_job(job_id: str) -> dict:
    with _lock:
        return _jobs.get(job_id, {})


def _set(job_id: str, **kwargs):
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


# ── Dataset discovery ──────────────────────────────────────────────────────────
# Template files that live in Dataset/ purely so /download_dataset_example can
# serve them — they demonstrate the schema, not real evaluation data, so they
# must never appear as a selectable evaluation dataset.
_TEMPLATE_FILENAMES = {"example_sheet.xlsx"}

# ── Quick vs Full evaluation — which sheets each one reads ──────────────────
# Quick Evaluation (mode=auto, split="demo") combines every sheet prefixed
# "Demo_" — a fast, small keyword-scored subset.
# Full Evaluation (mode=llm, split="all") combines every sheet prefixed
# "Dataset_" — this is genuinely the FULL question set the file ships (all
# Dataset_* sheets concatenated and deduped by "id", see _combine_sheets
# below), scored by the Groq LLM rubric instead of keyword matching. It is
# not a separate curated subset — "Dataset_*" IS the complete dataset.
# This is why example_sheet.xlsx's template sheets are named Demo_Quick_example
# (→ Quick) and Dataset_example (→ Full): the prefix is what run_evaluation()
# and list_available_datasets() actually key off of, so renaming a sheet to
# anything not starting with "Demo_"/"Dataset_" makes it invisible to both
# evaluation modes and to the dataset dropdown.


def list_available_datasets() -> list:
    """
    Scans DATASET_DIR (Dataset/) for .xlsx files usable as evaluation datasets —
    any file containing at least one Dataset_* or Demo_* sheet. Picks up future
    files automatically, no hardcoded filenames (except _TEMPLATE_FILENAMES).
    Returns [{filename, demo_sheets, dataset_sheets}], sorted by filename.
    """
    results = []
    for fname in sorted(os.listdir(DATASET_DIR)):
        if not fname.lower().endswith(".xlsx") or fname.startswith("~$"):
            continue
        if fname in _TEMPLATE_FILENAMES:
            continue
        try:
            sheets = pd.ExcelFile(os.path.join(DATASET_DIR, fname)).sheet_names
        except Exception:
            continue
        demo_sheets    = [s for s in sheets if s.startswith("Demo_")]
        dataset_sheets = [s for s in sheets if s.startswith("Dataset_")]
        if demo_sheets or dataset_sheets:
            results.append({
                "filename":       fname,
                "demo_sheets":    demo_sheets,
                "dataset_sheets": dataset_sheets,
            })
    return results


def _combine_sheets(xl: "pd.ExcelFile", prefix: str):
    """
    Concats every sheet whose name starts with `prefix` (in file order),
    dedup'ing by the 'id' column (keep last — later sheets are the updated ones).
    Returns (DataFrame, matched_sheet_names). Empty DataFrame if no match.
    """
    matched = [s for s in xl.sheet_names if s.startswith(prefix)]
    if not matched:
        return pd.DataFrame(), matched

    combined = pd.concat([xl.parse(s) for s in matched], ignore_index=True)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset="id", keep="last").reset_index(drop=True)
    return combined, matched


# ── Rubric ────────────────────────────────────────────────────────────────────
RUBRIC = {
    "legal_accuracy":      0.40,
    "citation_correct":    0.20,
    "retrieval_relevance": 0.20,
    "hallucination":       0.15,
    "clarity":             0.05,
}

RUBRIC_LABELS = {
    "legal_accuracy":      "Độ chính xác pháp lý",
    "citation_correct":    "Trích dẫn điều luật",
    "retrieval_relevance": "Mức độ liên quan ngữ cảnh",
    "hallucination":       "Kiểm soát bịa đặt",
    "clarity":             "Rõ ràng, dễ hiểu",
}


# ── Auto scorer (offline, fast) ───────────────────────────────────────────────
def _extract_article_numbers(article_ref: str) -> list:
    """Extract every 'Điều N[letter]' number named in a citation string.
    article_ref often carries extra numbers that must NOT be folded into the
    article number — a Khoản number ("Khoản 35 Điều 4"), a decree number/year
    ("Điều 17 Nghị định 168/2025/NĐ-CP"), or multiple articles joined by ";"
    ("Điều 27; Điều 38"). Matching only text that follows "Điều" keeps those
    separate instead of mashing every digit in the string into one number."""
    return re.findall(r'điều\s+(\d+[a-z]?)', article_ref, flags=re.IGNORECASE)


def _extract_law_numbers(article_ref: str) -> list:
    """Fallback for refs naming a law/decree with no 'Điều' component at all
    (e.g. "67/VBHN-VPQH 2025; Luật 76/2025/QH15") — match the doc number itself."""
    return re.findall(r'\d+/\d{4}/[A-ZĐ][A-ZĐ.\-]*', article_ref, flags=re.IGNORECASE)


def _auto_score(question: str, generated: str, expected: str,
                article_ref: str, keywords: str, retrieved_context: str) -> dict:
    gen_lower = generated.lower()
    exp_lower = expected.lower()
    scores    = {}

    art_nums = _extract_article_numbers(article_ref)
    if art_nums:
        cite_ok = any(re.search(rf'điều\s+{re.escape(n)}\b', gen_lower) for n in art_nums)
    else:
        law_nums = _extract_law_numbers(article_ref)
        cite_ok = any(n.lower() in gen_lower for n in law_nums) if law_nums else False
    scores["citation_correct"] = 3 if cite_ok else 0

    kw_list    = [k.strip().lower() for k in keywords.split(';') if k.strip()]
    kw_hits    = sum(1 for kw in kw_list if kw in gen_lower)
    kw_ratio   = kw_hits / max(len(kw_list), 1)
    exp_words  = set(exp_lower.split())
    gen_words  = set(gen_lower.split())
    exp_overlap = len(exp_words & gen_words) / max(len(exp_words), 1)
    scores["legal_accuracy"] = min(3, round((kw_ratio * 2 + exp_overlap) * 1.5))

    if retrieved_context:
        ctx_lower = retrieved_context.lower()
        ctx_hits  = sum(1 for kw in kw_list if kw in ctx_lower)
        scores["retrieval_relevance"] = min(3, round((ctx_hits / max(len(kw_list), 1)) * 3))
    else:
        scores["retrieval_relevance"] = 0

    if art_nums:
        correct_nums = set(art_nums)
        cited_arts   = re.findall(r'điều\s+(\d+[a-z]?)', gen_lower)
        wrong_arts   = [a for a in cited_arts if a not in correct_nums]
        scores["hallucination"] = 3 if len(wrong_arts) == 0 else (2 if len(wrong_arts) <= 1 else 1)
    else:
        # No "Điều N" in the reference to check cited numbers against (law/decree-
        # name-only citation) — can't reliably detect a wrong article here.
        scores["hallucination"] = 3

    word_count     = len(generated.split())
    scores["clarity"] = 3 if 30 <= word_count <= 200 else (2 if word_count >= 15 else 1)

    scores["total"] = round(
        sum((scores[k] / 3.0) * RUBRIC[k] * 100 for k in RUBRIC), 1
    )
    return scores


# ── LLM scorer (Groq) ─────────────────────────────────────────────────────────
def _llm_score(question: str, generated: str, expected: str,
               article_ref: str, groq_key: str) -> dict:
    from langchain_groq import ChatGroq
    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)

    prompt = f"""Bạn là giáo viên chấm điểm câu trả lời pháp lý.

Câu hỏi: {question}
Câu trả lời của AI:
{generated}
Câu trả lời mẫu:
{expected}
Điều luật cần trích dẫn: {article_ref}

Hãy chấm điểm từ 0-3 cho mỗi tiêu chí sau và trả về JSON:
- legal_accuracy: Độ chính xác pháp lý (0=sai, 1=thiếu, 2=cơ bản đúng, 3=đúng đầy đủ)
- citation_correct: Trích dẫn điều luật (0=không có/sai, 1=mơ hồ, 2=đúng chính, 3=đúng đầy đủ)
- retrieval_relevance: Nội dung dựa vào đúng điều luật (0-3)
- hallucination: Không bịa đặt (0=bịa nhiều, 1=có bịa, 2=ít bịa, 3=không bịa)
- clarity: Rõ ràng, dễ hiểu (0-3)

Chỉ trả về JSON, không giải thích. Ví dụ:
{{"legal_accuracy":2,"citation_correct":3,"retrieval_relevance":2,"hallucination":3,"clarity":3}}"""

    # Backoff: 15s, 30s, 60s, 90s, 120s
    wait_times = [15, 30, 60, 90, 120]
    for attempt in range(len(wait_times) + 1):
        try:
            response   = llm.invoke(prompt).content.strip()
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                sc = json.loads(json_match.group())
                sc["total"] = round(
                    sum((sc.get(k, 0) / 3.0) * RUBRIC[k] * 100 for k in RUBRIC), 1
                )
                return sc
        except Exception as e:
            if attempt < len(wait_times):
                wait = wait_times[attempt]
                print(f"  [429] Rate limit - retry {attempt + 1}/{len(wait_times)} after {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [FAIL] LLM scoring failed after all retries: {e}")
    return {k: 0 for k in RUBRIC} | {"total": 0.0}


def _prune_old_eval_files(keep: int = KEEP_LATEST_EVAL_FILES):
    """Keep only the `keep` most recently modified eval_results_*.xlsx files
    in BASE_DIR — older ones are just disk clutter with no UI to browse them."""
    files = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("eval_results_") and f.lower().endswith(".xlsx")
    ]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(BASE_DIR, f)), reverse=True)
    for f in files[keep:]:
        try:
            os.remove(os.path.join(BASE_DIR, f))
        except Exception:
            pass


def _save_latest_result(summary_payload: dict):
    try:
        with open(LATEST_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False)
    except Exception:
        pass


def _summary_from_xlsx(path: str) -> dict:
    """Recompute a summary_payload-shaped dict from an existing eval_results
    xlsx's per-question score_* columns — used to backfill a score card for a
    run that finished before eval_results_latest.json existed."""
    df_res = pd.read_excel(path)
    if df_res.empty or "score_total" not in df_res.columns:
        return None

    avg_scores = {}
    for k in RUBRIC:
        col = f"score_{k}"
        if col in df_res.columns:
            avg_scores[k] = round(df_res[col].mean(), 2)

    by_type = {}
    by_diff = {}
    if "question_type" in df_res.columns:
        for qt, grp in df_res.groupby("question_type"):
            by_type[qt] = round(grp["score_total"].mean(), 1)
    if "difficulty" in df_res.columns:
        for diff, grp in df_res.groupby("difficulty"):
            by_diff[diff] = round(grp["score_total"].mean(), 1)

    return {
        "total_questions": len(df_res),
        "avg_total":       round(df_res["score_total"].mean(), 1),
        "avg_scores":      avg_scores,
        "rubric_labels":   RUBRIC_LABELS,
        "rubric_weights":  RUBRIC,
        "by_type":         by_type,
        "by_difficulty":   by_diff,
        "output_file":     os.path.basename(path),
        "mode":            "auto",
        "split":           "",
        "dataset_file":    "",
        "sheets_used":     [],
    }


def get_latest_eval_result() -> dict:
    """Returns the persisted summary of the most recent evaluation run for the
    Evaluate tab to show on open. Falls back to reconstructing it from the
    newest eval_results_*.xlsx on disk (and persists that reconstruction) if
    no run has completed since eval_results_latest.json was introduced.
    Returns None if no evaluation result exists at all."""
    if os.path.exists(LATEST_RESULT_PATH):
        try:
            with open(LATEST_RESULT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    files = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("eval_results_") and f.lower().endswith(".xlsx")
    ]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(BASE_DIR, f)), reverse=True)

    summary = _summary_from_xlsx(os.path.join(BASE_DIR, files[0]))
    if summary:
        _save_latest_result(summary)
    return summary


# ── Main background task ──────────────────────────────────────────────────────
def run_evaluation(job_id: str, mode: str, split: str, dataset_file: str = None):
    """
    Evaluate the RAG system.
    mode         : "auto" (fast, keyword matching) | "llm" (Groq-based, accurate)
    split        : "demo" (all Demo_* sheets in the file) | "all"/"test" (all Dataset_* sheets)
    dataset_file : filename (not path) of an .xlsx in DATASET_DIR (Dataset/), as
                   returned by list_available_datasets(). Falls back to the
                   newest known dataset file when omitted.
    """
    if dataset_file:
        # Sanitize: filename only, no path traversal outside DATASET_DIR.
        xlsx_path = os.path.join(DATASET_DIR, os.path.basename(dataset_file))
    else:
        # Default to the 200-updated file (latest dataset); fall back to 150 if absent
        xlsx_path = os.path.join(DATASET_DIR, "enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx")
        if not os.path.exists(xlsx_path):
            xlsx_path = os.path.join(DATASET_DIR, "enterprise_law_full_rag_chatbot_dataset_150.xlsx")

    _set(job_id, status="running", message="Đang tải dataset…", progress=0, scores=None)

    try:
        if not os.path.exists(xlsx_path):
            _set(job_id, status="failed",
                 message="❌ Không tìm thấy file dataset. Vui lòng import dataset trước.")
            return

        dataset_label = os.path.basename(xlsx_path)
        xl = pd.ExcelFile(xlsx_path)

        if split == "demo":
            df, matched = _combine_sheets(xl, "Demo_")
            if df.empty:
                _set(job_id, status="failed",
                     message=f"❌ File '{dataset_label}' không có sheet Demo — không thể chạy Quick Evaluation cho file này.")
                return
        else:
            df, matched = _combine_sheets(xl, "Dataset_")
            if df.empty:
                _set(job_id, status="failed",
                     message=f"❌ File '{dataset_label}' không có sheet Dataset — không thể chạy Full Evaluation cho file này.")
                return
            if split == "test" and "split" in df.columns:
                df = df[df["split"] == "test"]

        n = len(df)
        _set(job_id, message=f"Bắt đầu đánh giá {n} câu hỏi (mode={mode}, split={split})…", progress=0)

        # Resolve Groq key for LLM mode
        groq_key = None
        if mode == "llm":
            key_path = os.path.join(BASE_DIR, "groqkey.txt")
            if os.path.exists(key_path):
                with open(key_path) as f:
                    groq_key = f.read().strip()
            if not groq_key:
                mode = "auto"
                _set(job_id, message="⚠️ Không có Groq key — chuyển sang auto mode…")

        from engine.rag_engine import ask_rag

        results      = []
        score_totals = {k: [] for k in RUBRIC}
        totals       = []

        for idx, (_, row) in enumerate(df.iterrows()):
            q_id       = str(row.get('id', f'Q{idx}')).strip()
            question   = str(row.get('question_vi', '')).strip()
            expected   = str(row.get('expected_answer_vi', '')).strip()
            art_ref    = str(row.get('article_reference', '')).strip()
            keywords   = str(row.get('retrieval_keywords', '')).strip()
            q_type     = str(row.get('question_type', '')).strip()
            difficulty = str(row.get('difficulty', '')).strip()

            if not question or question == 'nan':
                continue

            pct = round(((idx + 1) / n) * 100)
            _set(job_id,
                 message=f"[{idx + 1}/{n}] {question[:55]}…",
                 progress=pct)

            retrieved_context = ""
            try:
                result = ask_rag(question, return_debug=True)
                if isinstance(result, dict):
                    generated         = result["answer"]
                    retrieved_context = result.get("retrieved_context", "")
                else:
                    generated = str(result)
            except Exception as e:
                generated = f"ERROR: {e}"

            if mode == "llm" and groq_key:
                sc = _llm_score(question, generated, expected, art_ref, groq_key)
                time.sleep(3)   # ~20 req/min → stay under Groq free-tier limit
            else:
                sc = _auto_score(question, generated, expected, art_ref, keywords, retrieved_context)

            for k in RUBRIC:
                score_totals[k].append(sc.get(k, 0))
            totals.append(sc["total"])

            results.append({
                "id":            q_id,
                "question_type": q_type,
                "difficulty":    difficulty,
                "question":      question,
                "generated":     generated,
                "expected":      expected,
                "article_ref":   art_ref,
                **{f"score_{k}": sc.get(k, 0) for k in RUBRIC},
                "score_total":   sc["total"],
            })

        avg_total  = round(sum(totals) / max(len(totals), 1), 1)
        avg_scores = {
            k: round(sum(score_totals[k]) / max(len(score_totals[k]), 1), 2)
            for k in RUBRIC
        }

        df_res  = pd.DataFrame(results)
        by_type = {}
        by_diff = {}
        if "question_type" in df_res.columns:
            for qt, grp in df_res.groupby("question_type"):
                by_type[qt] = round(grp["score_total"].mean(), 1)
        if "difficulty" in df_res.columns:
            for diff, grp in df_res.groupby("difficulty"):
                by_diff[diff] = round(grp["score_total"].mean(), 1)

        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_stem = os.path.splitext(dataset_label)[0]
        out_path     = os.path.join(BASE_DIR, f"eval_results_{dataset_stem}_{split}_{mode}_{ts}.xlsx")
        df_res.to_excel(out_path, index=False)
        _prune_old_eval_files()

        summary_payload = {
            "total_questions": len(results),
            "avg_total":       avg_total,
            "avg_scores":      avg_scores,
            "rubric_labels":   RUBRIC_LABELS,
            "rubric_weights":  RUBRIC,
            "by_type":         by_type,
            "by_difficulty":   by_diff,
            "output_file":     os.path.basename(out_path),
            "mode":            mode,
            "split":           split,
            "dataset_file":    dataset_label,
            "sheets_used":     matched,
        }
        _save_latest_result(summary_payload)

        _set(job_id,
             status="done",
             message=f"✅ Đánh giá hoàn tất! Điểm tổng: {avg_total}/100",
             progress=100,
             scores=summary_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set(job_id, status="failed", message=f"❌ Lỗi: {e}", scores=None)


# ── CLI entrypoint (with tqdm progress bar) ───────────────────────────────────
def _run_cli(mode: str, split: str):
    """Interactive terminal run with live progress bar and running score."""
    from tqdm import tqdm

    xlsx_path = os.path.join(DATASET_DIR, "enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx")
    if not os.path.exists(xlsx_path):
        xlsx_path = os.path.join(DATASET_DIR, "enterprise_law_full_rag_chatbot_dataset_150.xlsx")
    if not os.path.exists(xlsx_path):
        print("[ERROR] Dataset file not found. Run build_db_from_dataset.py first.")
        sys.exit(1)

    xl     = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names

    demo_sheet = next((s for s in ["Demo_50", "Demo_30"] if s in sheets), None)
    if split == "demo" and demo_sheet:
        df = xl.parse(demo_sheet)
    else:
        ds_sheet = next(
            (s for s in ["Dataset_200", "Dataset_150"] if s in sheets),
            next((s for s in sheets if s.startswith("Dataset_")), None)
        )
        if ds_sheet is None:
            print("[ERROR] No Dataset_* sheet found in file.")
            sys.exit(1)
        df = xl.parse(ds_sheet)
        if split == "test" and "split" in df.columns:
            df = df[df["split"] == "test"]

    n = len(df)

    # Resolve Groq key
    groq_key = None
    if mode == "llm":
        key_path = os.path.join(BASE_DIR, "groqkey.txt")
        if os.path.exists(key_path):
            with open(key_path) as f:
                groq_key = f.read().strip()
        if not groq_key:
            print("[!] groqkey.txt not found — switching to auto mode")
            mode = "auto"

    print(f"\nEvaluating {n} questions  |  mode={mode}  |  split={split}")
    if mode == "llm":
        est = n * 5 // 60
        print(f"Estimated time: ~{est}-{est*2} minutes (LLM scoring + 3s throttle)\n")
    else:
        print()

    from engine.rag_engine import ask_rag

    results      = []
    score_totals = {k: [] for k in RUBRIC}
    totals       = []

    with tqdm(total=n, ncols=90, unit="q") as pbar:

        for idx, (_, row) in enumerate(df.iterrows()):
            q_id       = str(row.get("id",            f"Q{idx}")).strip()
            question   = str(row.get("question_vi",   "")).strip()
            expected   = str(row.get("expected_answer_vi", "")).strip()
            art_ref    = str(row.get("article_reference",  "")).strip()
            keywords   = str(row.get("retrieval_keywords", "")).strip()
            q_type     = str(row.get("question_type", "")).strip()
            difficulty = str(row.get("difficulty",    "")).strip()

            if not question or question == "nan":
                pbar.update(1)
                continue

            # ── Step 1: RAG answer ──
            pbar.set_description(f"[{q_id}] RAG...")
            retrieved_context = ""
            try:
                result = ask_rag(question, return_debug=True)
                if isinstance(result, dict):
                    generated         = result["answer"]
                    retrieved_context = result.get("retrieved_context", "")
                else:
                    generated = str(result)
            except Exception as e:
                generated = f"ERROR: {e}"

            # ── Step 2: Score ──
            pbar.set_description(f"[{q_id}] Scoring...")
            if mode == "llm" and groq_key:
                sc = _llm_score(question, generated, expected, art_ref, groq_key)
                time.sleep(3)
            else:
                sc = _auto_score(question, generated, expected, art_ref, keywords, retrieved_context)

            for k in RUBRIC:
                score_totals[k].append(sc.get(k, 0))
            totals.append(sc["total"])

            results.append({
                "id":            q_id,
                "question_type": q_type,
                "difficulty":    difficulty,
                "question":      question,
                "generated":     generated,
                "expected":      expected,
                "article_ref":   art_ref,
                **{f"score_{k}": sc.get(k, 0) for k in RUBRIC},
                "score_total":   sc["total"],
            })

            running_avg = sum(totals) / len(totals)
            pbar.set_description(f"[{q_id}]")
            pbar.set_postfix(score=f"{running_avg:.1f}/100", last=f"{sc['total']:.0f}")
            pbar.update(1)

    # ── Summary ──
    avg_total  = round(sum(totals) / max(len(totals), 1), 1)
    avg_scores = {k: round(sum(score_totals[k]) / max(len(score_totals[k]), 1), 2) for k in RUBRIC}

    df_res  = pd.DataFrame(results)
    by_type = {}
    by_diff = {}
    if "question_type" in df_res.columns:
        for qt, grp in df_res.groupby("question_type"):
            by_type[qt] = round(grp["score_total"].mean(), 1)
    if "difficulty" in df_res.columns:
        for d, grp in df_res.groupby("difficulty"):
            by_diff[d] = round(grp["score_total"].mean(), 1)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(BASE_DIR, f"eval_results_{split}_{mode}_{ts}.xlsx")
    df_res.to_excel(out_path, index=False)

    W = 55
    print(f"\n{'='*W}")
    print(f"  TONG DIEM : {avg_total}/100   ({len(results)} cau, {mode} mode)")
    print(f"{'='*W}")
    for k, avg in avg_scores.items():
        label  = RUBRIC_LABELS.get(k, k)
        weight = RUBRIC.get(k, 0) * 100
        bar_w  = int((avg / 3.0) * 20)
        bar    = "[" + "#" * bar_w + "." * (20 - bar_w) + "]"
        print(f"  {label:<35} {bar} {avg:.2f}/3  ({weight:.0f}%)")

    if by_type:
        print(f"\n  Theo loai cau hoi:")
        for qt, v in sorted(by_type.items()):
            fill = int(v / 5)
            print(f"    {qt:<22} {'|'*fill} {v}/100")

    if by_diff:
        print(f"\n  Theo do kho:")
        for d, v in sorted(by_diff.items()):
            fill = int(v / 5)
            print(f"    {d:<10} {'|'*fill} {v}/100")

    print(f"\n  Ket qua : {os.path.basename(out_path)}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate RAG system (standalone)")
    parser.add_argument("--mode",  choices=["auto", "llm"], default="auto",
                        help="auto = fast keyword scoring | llm = Groq LLM scoring")
    parser.add_argument("--split", choices=["demo", "all", "test"], default="demo",
                        help="demo = 30 questions | all = full dataset | test = test split")
    args = parser.parse_args()

    _run_cli(mode=args.mode, split=args.split)
