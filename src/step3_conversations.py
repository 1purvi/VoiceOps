# %% [markdown]
# # Step 3: hold a conversation (tools + memory)
# Agent = `agent.Agent`. Each customer message runs a loop: LLM → tools → LLM → `final_answer`.
# Tools: `search_kb` (Step 2 retrieval, k = TOP_K), `lookup_order` (identity-checked), `create_ticket`,
# `date_calc`. Results are cached per conversation in `outputs/cache_conv_<RUN_TAG>_<model>.json`.

# %%
import pandas as pd
import agent as ct
from agent import Agent, run_conversations, score_conversations, conv_summary, load_conversations

pd.set_option("display.max_colwidth", 200)
print("RUN_TAG:", ct.RUN_TAG, "| search k:", ct.SEARCH_K, "| model:", ct.LLM_MODEL)

# %%
# Tools work without the LLM (no tokens): verification boundary + business days
conv = load_conversations()
oid = conv[conv.order_id != ""].order_id.iloc[0]
print("unverified :", ct._lookup_order(oid))                                  # must contain no order details
print("wrong value:", ct._lookup_order(oid, verification_value="0000"))
print("date_calc  :", ct.date_calc("2026-10-17", 7, "business"))             # 27 Oct (skips Sun 18, holiday 19)

# %% [markdown]
# ## Try it: the example conversation from the brief
# Replace the order ID and verification value with a real pair from `orders.csv` / `customers.csv` to see the
# verified path. Each `turn` costs a few LLM calls.

# %%
def chat(agent, text):
    out = agent.turn(text)
    tools = [(c["name"], {k: v for k, v in c.items() if k in ("query", "order_id", "verified", "ticket_id",
                                                               "result_date")})
             for c in out["tool_calls"]]
    print(f"CUSTOMER: {text}\nAGENT [{out['decision']}]: {out['reply']}\n  tools: {tools}\n"
          f"  sections: {out['used_sections']}\n")
    return out


a = Agent()
chat(a, "My order still hasn't arrived. Can I cancel it?")
chat(a, f"The order ID is {oid}")
chat(a, "How long will the refund take?")
chat(a, "I want ₹500 compensation right now.")

# ## Run the 23 practice conversations
# Run 2 first to check the output, then all. Re-running only retries conversations that failed (API errors).
# Delete `tools/tickets_created.csv` before the final run; every run appends tickets to it.

# %%
run_conversations(limit=2)

# %%
run_conversations()

# %% [markdown]
# ## Metrics
# A turn is correct when the decision matches, every must_include fact is present, there is no
# must_not_include phrase, and every expected tool call was made.

df = score_conversations()
if df.empty:
    raise SystemExit("No finished conversations to score yet (e.g. Groq daily quota). Rerun run_conversations().")
report = pd.Series(conv_summary(df), name=f"Step 3 ({ct.RUN_TAG})")
report.to_csv(ct.OUT_DIR / f"step3_metrics_{ct.RUN_TAG}.csv")
print(report.to_string(), "\n")
for col in ("expected", "language", "scenario"):
    print(df.groupby(col).agg(turns=("turn_correct", "size"), correct=("turn_correct", "sum"),
                              decision_ok=("decision_ok", "mean"), tools_ok=("tools_ok", "mean"))
          .round(2).to_string(), "\n")

# %%
# Failures with their cause
fail = df[~df.turn_correct].copy()
fail["cause"] = fail.apply(lambda r: "wrong decision" if not r.decision_ok
                           else "tool calls" if not r.tools_ok
                           else "violation" if r.violation else "fact missing", axis=1)
print(fail.cause.value_counts().to_string(), "\n")
for _, r in fail.iterrows():
    print(f"{r.conversation_id} t{r.turn} [{r.language}] {r.cause} | expected={r.expected} got={r.decision}")
    if not r.tools_ok:
        print(f"  expected tools: {r.expected_tools}\n  actual tools:   "
              f"{[(t['name'], t.get('order_id'), t.get('verification')) for t in r.actual_tools]}")
    if r.missing or r.violations:
        print(f"  missing: {r.missing}  violations: {r.violations}")
    print(f"  reply: {r.reply}\n")
fail.to_csv(ct.OUT_DIR / f"step3_failures_{ct.RUN_TAG}.csv", index=False)

# %%
# Read one whole conversation: customer, expected behaviour, agent reply, tools
def show(cid):
    cache = ct._load_cache().get(cid, {})
    for _, row in conv[conv.conversation_id == cid].sort_values("turn", key=lambda s: s.astype(int)).iterrows():
        t = next((x for x in cache.get("turns", []) if str(x["turn"]) == row.turn), {})
        print(f"--- turn {row.turn} | expected {row.expected_decision} {row.expected_tool_calls}")
        print(f"CUSTOMER: {row.customer_message}")
        print(f"AGENT [{t.get('decision')}]: {t.get('reply')}")
        print(f"  tools: {[(c['name'], c.get('order_id'), c.get('verified'), c.get('ticket_id')) for c in t.get('tool_calls', []) if c['name'] != 'search_kb']}")
        print(f"  notes: {row.notes}\n")


show(fail.conversation_id.iloc[0] if len(fail) else conv.conversation_id.iloc[0])
