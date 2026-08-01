# evaluate_rag.py
# Evaluates the RAG system using enterprise_law_full_rag_chatbot_dataset_150.xlsx
#
# Evaluation rubric (from Evaluation_Rubric sheet):
#   Legal accuracy          40% — correct legal rule, right conditions
#   Citation correctness    20% — cites the right Điều
#   Retrieval relevance     20% — retrieved the right context
#   Hallucination control   15% — no fabricated content
#   Clarity                  5% — clear and educational
#
# Modes:
#   --mode auto   : fast automatic scoring (keyword + citation matching)
#   --mode llm    : use Groq LLM to score each answer (slower, more accurate)
#   --split test  : evaluate test set (120 questions, default)
#   --split demo  : evaluate demo set (30 questions)
#   --split all   : evaluate all 150 questions
#
# USAGE:
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python evaluate_rag.py --mode auto --split demo
#   python evaluate_rag.py --mode llm  --split test

import os
import sys
import json
import argparse
import time
import re
import warnings

# Silence two known-harmless, third-party UserWarnings that otherwise clutter
# every eval run: (1) gdown 4.4.0 (a transitive dep of vietocr, used for PDF
# OCR in import_law_engine.py/build_db_from_pdf.py) still imports the
# deprecated pkg_resources API; (2) openpyxl warns when a .xlsx has Excel
# conditional-formatting/data-validation rules it doesn't parse — the actual
# cell data is read correctly regardless, only that cosmetic formatting is
# dropped. Registered before pandas/engine imports below so it's in effect
# for every downstream import that might trigger them.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Conditional Formatting extension.*")
warnings.filterwarnings("ignore", message=".*Data Validation extension.*")

import pandas as pd
from datetime import datetime
from langchain_groq import ChatGroq

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # evaluate/ → root
XLSX_PATH = os.path.join(BASE_DIR, "enterprise_law_full_rag_chatbot_dataset_150.xlsx")

# Add project root to path so engine.rag_engine resolves correctly
# regardless of which directory the script is run from
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.rag_engine import ask_rag

# =========================
# RUBRIC WEIGHTS
# =========================
RUBRIC = {
    "legal_accuracy":     0.40,
    "citation_correct":   0.20,
    "retrieval_relevance":0.20,
    "hallucination":      0.15,
    "clarity":            0.05,
}


# =========================
# AUTO SCORER
# =========================
def auto_score(question: str, generated: str, expected: str,
               article_ref: str, keywords: str, retrieved_context: str) -> dict:
    """
    Fast automatic scoring using keyword and citation matching.
    Not as accurate as LLM scoring but works offline instantly.
    """
    gen_lower  = generated.lower()
    exp_lower  = expected.lower()
    art_lower  = article_ref.lower().replace(" ", "")

    scores = {}

    # ── Citation correctness (0-3)
    # Check if generated answer mentions the correct Điều
    art_num = re.sub(r'[^\d]', '', article_ref)
    citation_patterns = [
        f"điều {art_num}",
        f"điều {art_num}.",
        f"điều {art_num},",
        art_lower,
    ]
    citation_found = any(p in gen_lower for p in citation_patterns)
    scores["citation_correct"] = 3 if citation_found else 0

    # ── Keyword overlap (proxy for legal accuracy)
    kw_list = [k.strip().lower() for k in keywords.split(';') if k.strip()]
    kw_hits = sum(1 for kw in kw_list if kw in gen_lower)
    kw_ratio = kw_hits / max(len(kw_list), 1)

    # Also check expected answer overlap
    exp_words  = set(exp_lower.split())
    gen_words  = set(gen_lower.split())
    exp_overlap = len(exp_words & gen_words) / max(len(exp_words), 1)

    accuracy_score = min(3, round((kw_ratio * 2 + exp_overlap) * 1.5))
    scores["legal_accuracy"] = accuracy_score

    # ── Retrieval relevance (0-3)
    if retrieved_context:
        ctx_lower = retrieved_context.lower()
        ctx_kw_hits = sum(1 for kw in kw_list if kw in ctx_lower)
        ctx_ratio = ctx_kw_hits / max(len(kw_list), 1)
        scores["retrieval_relevance"] = min(3, round(ctx_ratio * 3))
    else:
        scores["retrieval_relevance"] = 0

    # ── Hallucination control (0-3) — penalize if fabricated Điều cited
    # Look for article numbers in response that aren't the expected one
    cited_arts = re.findall(r'điều\s+(\d+)', gen_lower)
    wrong_arts = [a for a in cited_arts if a != art_num]
    if len(wrong_arts) == 0:
        scores["hallucination"] = 3
    elif len(wrong_arts) <= 1:
        scores["hallucination"] = 2
    else:
        scores["hallucination"] = 1

    # ── Clarity (0-3) — proxy: response length and structure
    word_count = len(generated.split())
    if word_count >= 30 and word_count <= 200:
        scores["clarity"] = 3
    elif word_count >= 15:
        scores["clarity"] = 2
    else:
        scores["clarity"] = 1

    # ── Weighted total (0-100)
    total = sum(
        (scores[k] / 3.0) * RUBRIC[k] * 100
        for k in RUBRIC
    )
    scores["total"] = round(total, 1)

    return scores


# =========================
# LLM SCORER
# =========================
def llm_score(question: str, generated: str, expected: str,
              article_ref: str, groq_api_key: str = None) -> dict:
    """
    Use Groq LLM to evaluate quality against the rubric.
    Returns scores dict.
    """
    from engine.groq_keys import current_key
    llm = ChatGroq(api_key=groq_api_key or current_key(), model="llama-3.1-8b-instant", temperature=0)

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
- retrieval_relevance: Nội dung có dựa vào đúng điều luật không (0-3)
- hallucination: Không bịa đặt (0=bịa nhiều, 1=có bịa, 2=ít bịa, 3=không bịa)
- clarity: Rõ ràng, dễ hiểu (0-3)

Chỉ trả về JSON, không giải thích. Ví dụ:
{{"legal_accuracy":2,"citation_correct":3,"retrieval_relevance":2,"hallucination":3,"clarity":3}}"""

    try:
        response = llm.invoke(prompt).content.strip()
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            scores = json.loads(json_match.group())
            total = sum(
                (scores.get(k, 0) / 3.0) * RUBRIC[k] * 100
                for k in RUBRIC
            )
            scores["total"] = round(total, 1)
            return scores
    except Exception as e:
        print(f"  LLM scoring error: {e}")

    return {k: 0 for k in RUBRIC} | {"total": 0.0}


# =========================
# MAIN EVALUATION LOOP
# =========================
def run_evaluation(split: str, mode: str, max_questions: int = None):

    # Load data
    xl = pd.ExcelFile(XLSX_PATH)
    if split == "all":
        df = xl.parse('Dataset_150')
    elif split == "demo":
        df = xl.parse('Demo_30')
    else:
        df = xl.parse('Dataset_150')
        df = df[df['split'] == 'test']

    if max_questions:
        df = df.head(max_questions)

    print(f"\n{'='*60}")
    print(f"RAG Evaluation — split={split}, mode={mode}, n={len(df)}")
    print(f"{'='*60}\n")

    # Load Groq key(s) for LLM mode — groqkey.txt may hold several,
    # semicolon-separated (see engine/groq_keys.py)
    groq_key = None
    if mode == "llm":
        from engine.groq_keys import get_keys
        keys = get_keys()
        if keys:
            groq_key = keys[0]
        else:
            print("⚠️  groqkey.txt not found — falling back to auto mode")
            mode = "auto"

    results = []
    score_totals = {k: [] for k in RUBRIC}
    totals = []

    for i, row in df.iterrows():
        q_id       = str(row.get('id', f'Q{i}')).strip()
        question   = str(row.get('question_vi', '')).strip()
        expected   = str(row.get('expected_answer_vi', '')).strip()
        article_ref = str(row.get('article_reference', '')).strip()
        keywords   = str(row.get('retrieval_keywords', '')).strip()
        q_type     = str(row.get('question_type', '')).strip()
        difficulty = str(row.get('difficulty', '')).strip()

        if not question or question == 'nan':
            continue

        print(f"[{q_id}] ({difficulty}/{q_type}) {question[:60]}...")

        # Generate answer via RAG
        t0 = time.time()
        retrieved_context = ""   # default — prevents NameError if ask_rag throws
        try:
            rag_result = ask_rag(question, return_debug=True)
            if isinstance(rag_result, dict):
                generated = rag_result["answer"]
                retrieved_context = rag_result.get("retrieved_context", "")
            else:
                generated = str(rag_result)
        except Exception as e:
            generated = f"ERROR: {e}"
        elapsed = round(time.time() - t0, 2)

        # Score
        if mode == "llm" and groq_key:
            scores = llm_score(question, generated, expected, article_ref, groq_key)
        else:
            # For auto mode, use empty retrieved context (not available at this level)
            scores = auto_score(
                question,
                generated,
                expected,
                article_ref,
                keywords,
                retrieved_context
            )

        print(f"  Score: {scores['total']:.1f}/100 | "
              f"acc={scores['legal_accuracy']}/3 "
              f"cite={scores['citation_correct']}/3 "
              f"hall={scores['hallucination']}/3 "
              f"({elapsed}s)")

        for k in RUBRIC:
            score_totals[k].append(scores.get(k, 0))
        totals.append(scores["total"])

        results.append({
            "id":            q_id,
            "question_type": q_type,
            "difficulty":    difficulty,
            "question":      question,
            "generated":     generated,
            "expected":      expected,
            "article_ref":   article_ref,
            "elapsed_s":     elapsed,
            **{f"score_{k}": scores.get(k, 0) for k in RUBRIC},
            "score_total":   scores["total"],
        })

    # =========================
    # SUMMARY REPORT
    # =========================
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — {len(results)} questions")
    print(f"{'='*60}")
    print(f"{'Metric':<28} {'Avg/3':>8}  {'Weighted %':>12}")
    print(f"{'-'*50}")
    for k, weight in RUBRIC.items():
        avg = sum(score_totals[k]) / max(len(score_totals[k]), 1)
        weighted = (avg / 3.0) * weight * 100
        print(f"  {k:<26} {avg:>6.2f}/3   {weighted:>8.1f}%")
    print(f"{'-'*50}")
    avg_total = sum(totals) / max(len(totals), 1)
    print(f"  {'TOTAL SCORE':<26} {avg_total:>8.1f}/100")
    print(f"{'='*60}")

    # By question type
    df_res = pd.DataFrame(results)
    if 'question_type' in df_res.columns:
        print("\nBy question type:")
        for qt, grp in df_res.groupby('question_type'):
            print(f"  {qt:<20} avg={grp['score_total'].mean():.1f}/100  n={len(grp)}")
    if 'difficulty' in df_res.columns:
        print("\nBy difficulty:")
        for diff, grp in df_res.groupby('difficulty'):
            print(f"  {diff:<10} avg={grp['score_total'].mean():.1f}/100  n={len(grp)}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(BASE_DIR, f"eval_results_{split}_{mode}_{ts}.xlsx")
    df_res.to_excel(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    return df_res


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--mode",  choices=["auto","llm"], default="auto",
                        help="Scoring mode: auto (fast) or llm (accurate)")
    parser.add_argument("--split", choices=["test","demo","all"], default="demo",
                        help="Dataset split to evaluate")
    parser.add_argument("--max",   type=int, default=None,
                        help="Max questions to evaluate (for quick test)")
    args = parser.parse_args()

    run_evaluation(split=args.split, mode=args.mode, max_questions=args.max)