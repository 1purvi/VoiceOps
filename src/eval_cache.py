"""Run practice questions with a per-question cache.
- Every result is saved with llm_status = success | failed (from the llm_ok flag, not the reply text).
- Re-running skips successes and only calls the LLM for failed or missing questions.
- Scoring is separate, so changing metrics.py never costs LLM tokens."""
import json

import pandas as pd

from config import OUT_DIR, PRACTICE_DIR, LLM_MODEL
from metrics import score, split, retrieval_metrics


def load_questions():
    return pd.read_csv(PRACTICE_DIR / "questions.csv", dtype=str, keep_default_na=False)


def cache_path(mode):
    return OUT_DIR / f"cache_{mode}_{LLM_MODEL.replace('/', '_')}.json"


def load_cache(mode):
    p = cache_path(mode)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_cache(mode, cache):
    cache_path(mode).write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def run_cached(mode, answer_fn, questions=None, limit=None):
    """mode: 'baseline' or 'rag'. answer_fn(question_text) -> dict with llm_ok."""
    q = load_questions() if questions is None else questions
    if limit:
        q = q.head(limit)
    cache = load_cache(mode)
    todo = [r for _, r in q.iterrows() if cache.get(r.id, {}).get("llm_status") != "success"]
    print(f"{mode}: {len(q) - len(todo)} cached, {len(todo)} to run")
    for row in todo:
        try:
            out = answer_fn(row.question)
        except Exception as e:                                   # retrieval crash etc.
            out = {"llm_ok": False, "error": f"{type(e).__name__}: {e}"[:300], "reply": ""}
        out["llm_status"] = "success" if out.get("llm_ok") else "failed"
        cache[row.id] = out
        save_cache(mode, cache)                                  # save after every question
        print(f"  {row.id}: {out['llm_status']:7s} {out.get('decision') or out.get('error', '')}")
    n_failed = sum(1 for r in q.id if cache.get(r, {}).get("llm_status") != "success")
    print(f"{mode}: done — {n_failed} still failed" + (" (rerun to retry them)" if n_failed else ""))
    return cache


def score_cached(mode, questions=None):
    """Score everything in the cache. No LLM calls."""
    q = load_questions() if questions is None else questions
    cache = load_cache(mode)
    rows = []
    for _, row in q.iterrows():
        out = cache.get(row.id)
        if not out or out.get("llm_status") != "success":
            continue                                              # API failures are not scored as wrong
        gold = split(row.gold_evidence)
        rows.append({"id": row.id, "language": row.language, "intent": row.intent,
                     "expected": row.expected_decision, "decision": out.get("decision"),
                     "json_ok": out.get("json_ok"), "reply": out.get("reply", ""),
                     "retrieved": out.get("retrieved", []), "used_sections": out.get("used_sections", []),
                     "gold": gold}
                    | score(row, out)
                    | (retrieval_metrics(out["retrieved"], gold, out.get("used_sections"))
                       if "retrieved" in out else {}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"results_{mode}.csv", index=False)
    return df


def summary(df):
    s = {"scored": len(df),
         "correct": f"{int(df.correct.sum())} / {len(df)} ({df.correct.mean():.1%})",
         "decision accuracy": f"{df.decision_ok.mean():.1%}",
         "avg fact recall": f"{df.fact_recall.mean():.1%}",
         "violations": int(df.violation.sum())}
    for col in ("recall@k", "hit@k", "mrr", "cite_precision"):
        if col in df:
            s[col] = round(float(df[col].mean()), 3)
    return s


def clear_cache(mode):
    """Call after changing the prompt, chunking, TOP_K or retrieval."""
    p = cache_path(mode)
    if p.exists():
        p.unlink(); print("deleted", p.name)
