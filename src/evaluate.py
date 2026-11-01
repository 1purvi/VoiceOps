"""Steps 1-3 evaluation. run.py calls run_all() before it opens the app.

  Step 1  LLM only        practice/questions.csv  -> results_baseline.csv
  Step 2  LLM + RAG       practice/questions.csv  -> results_rag_v4.csv, step1_vs_step2.csv
  Step 3  agent + tools   practice/conversations.csv -> results_conversations_v4.csv,
                                                         step3_metrics_v4.csv, step3_failures_v4.csv

Every LLM answer is cached in outputs/ (cache_*.json). A second run only scores the cached answers, so it
takes seconds; run.py --fresh deletes the caches and asks the LLM again. All files are written to outputs/.
"""
import pandas as pd

from config import OUT_DIR

RAG_MODE = "rag_v4"


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def clear_caches():
    """Delete every cached answer so Steps 1-3 call the LLM again."""
    import agent
    from eval_cache import cache_path
    for p in (cache_path("baseline"), cache_path(RAG_MODE), agent.conv_cache_path()):
        if p.exists():
            p.unlink()
            print("deleted", p.name)


def step1():
    from eval_cache import run_cached, score_cached, summary
    from llm import ask
    banner("STEP 1: LLM only (no documents) on practice/questions.csv")
    run_cached("baseline", ask)
    base = score_cached("baseline")
    print(pd.Series(summary(base)).to_string())
    return base


def step2(base):
    from eval_cache import run_cached, score_cached, summary
    from rag_pipeline import rag_answer
    banner("STEP 2: LLM + RAG over kb/ on practice/questions.csv")
    run_cached(RAG_MODE, rag_answer)
    rag = score_cached(RAG_MODE)
    table = pd.DataFrame({"Step 1 (no docs)": summary(base), "Step 2 (RAG)": summary(rag)})
    table.to_csv(OUT_DIR / "step1_vs_step2.csv")
    print(table.to_string())
    return rag


def step3():
    import agent
    banner("STEP 3: conversation agent (tools + memory) on practice/conversations.csv")
    agent.run_conversations()
    df = agent.score_conversations()
    if df.empty:
        print("No conversation finished (see the errors above). Run again to retry; finished ones stay cached.")
        return df
    report = pd.Series(agent.conv_summary(df), name=f"Step 3 ({agent.RUN_TAG})")
    report.to_csv(OUT_DIR / f"step3_metrics_{agent.RUN_TAG}.csv")
    print(report.to_string())
    for col in ("expected", "language", "scenario"):
        print("\n" + df.groupby(col).agg(turns=("turn_correct", "size"), correct=("turn_correct", "sum"),
                                         decision_ok=("decision_ok", "mean"), tools_ok=("tools_ok", "mean"))
              .round(2).to_string())
    fail = df[~df.turn_correct].copy()
    if len(fail):
        fail["cause"] = fail.apply(lambda r: "wrong decision" if not r.decision_ok
                                   else "tool calls" if not r.tools_ok
                                   else "violation" if r.violation else "fact missing", axis=1)
        print("\nfailures:\n" + fail.cause.value_counts().to_string())
    fail.to_csv(OUT_DIR / f"step3_failures_{agent.RUN_TAG}.csv", index=False)
    return df


def run_all(fresh=False):
    """Run Steps 1, 2 and 3 in order and save every result in outputs/."""
    if fresh:
        clear_caches()
    base = step1()
    step2(base)
    step3()
    banner(f"Evaluation finished. Results saved in {OUT_DIR}")
