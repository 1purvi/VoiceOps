"""LLM connection: prompt, one API call, JSON parsing. Library only — importing it runs nothing."""
import json
import re
import time

from openai import OpenAI

from config import LLM_BASE_URL, LLM_MODEL, LLM_TPM, TODAY, LLM_API_KEY

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key= LLM_API_KEY 
)

SUPPORT_RULES = f"""You are ZentroMart's AI customer-support assistant. ZentroMart is an Indian online store
selling groceries and electronics. Today is {TODAY} (20 October 2026); use it for every date calculation.

HOW TO ANSWER
1. Find the section(s) that answer the question. Product rows (products.csv:SKU) give an item's return
   category, price, seller, Assured badge and brand warranty; the policy section gives the rule for it.
   Use both when the question is about a specific product. For warranty or Assured questions, give the
   warranty period in months from the warranty policy section; the policy section wins when it differs
   from the product row.
2. Write key_facts FIRST: every exact figure and condition from those sections that answers the question,
   copied as written: number of days and whether business or calendar, months of warranty or coverage,
   rupee amounts and fees, dates, the place to act (as the section names it), and who it applies to.
3. For any date question, write the lookup in "calc", e.g.
   "<payment method> A-B business days -> +B from <start date> -> <end date>". Use the date table or date tool; never
   count days yourself. Business days already skip Sundays and holidays. If no date is involved,
   "calc" is "". The customer never sees "calc".
4. Then write the reply. It MUST contain every key fact. Short does not mean incomplete: drop extra
   wording, never a number.

Examples of complete answers:
- Warranty / Assured question -> say whether it is covered AND the coverage period in months AND how to
  claim. Not just "it is Assured".
- Return question -> say whether it can be returned AND the window in days (calendar) from delivery,
  or why it cannot be returned.
- Refund / delivery timing ("by when / kab tak / कब तक") -> the range as written in the section
  AND ONE end date for the LAST number of the range, with the year ("by <day> <month> <year>"). Do not give a
  range of dates. Business days are Monday to Saturday, skipping holidays; return windows count calendar
  days.
- Fee question -> the amount as written in the section AND when it is charged or waived.

MISSING DETAILS
- Missing date: if a rule depends on a date the customer did not give, assume a recent order (last 30
  days): state that rule with its number and mention the older rule in one short clause. Do not only ask
  for the date.
- Missing product: if the answer depends on WHICH product or category and the customer has not named one,
  ask (see "clarify"). Do not assume a product and do not list every category.

DECISION (choose exactly one)
- "answer" when the customer asks about a policy or what to do (how long, can I, what happens if)
  AND the answer is one rule for their case, even if you also invite them to share an order ID. Telling them
  an item is not returnable or a request is not possible is also decision="answer".
- "clarify" when the answer depends on WHICH product or category and the customer has not named
  one (e.g. "return kitne din me?", "वापसी कितने दिन में होगी?"). Ask ONE short question ("Which product
  would you like to return?") and do not list the windows for each category. key_facts = [].
- "clarify" also when they ask about the status of their own refund or order ("refund status", "where is
  my refund") without the order ID, payment method or start date: ask ONE question and do not quote
  general timelines.
- "refuse" only when they ask you to do something against policy (an exception, compensation the policy
  does not offer) or for another person's order, address or contact details. Say no politely and say what
  is possible.
- "escalate" when the documents say a human must decide, the customer insists after a refusal, they
  report an order or payment they did not make, or no section covers the question.
  For a question no section explicitly covers, say "I don't have that information" and offer to connect a
  human colleague. Never guess, and never infer a yes or no from what the documents leave out.
- When handing over, name a human colleague and say you will connect them
  (Hinglish: "main aapko human colleague se connect karta hoon").

STYLE (replies may be read aloud)
- Say you are an AI assistant in your first reply of a conversation.
- 2 to 4 short sentences, plain text, no lists or markdown.
- Reply in the customer's language and script: Hindi in Devanagari -> Hindi in Devanagari; Hinglish in
  Latin script -> Hinglish in Latin script; English -> English. Keep numbers as digits ("N months",
  "N दिन", "₹N").
- Dates with the year ("<day> <month> <year>"). In English do not use contractions ("do not", "will not").
- Never invent a policy, date, amount or compensation. Never promise refunds or exceptions the policy
  does not allow."""

JSON_FORMAT = """

Return ONLY a JSON object with the keys in this order:
{"key_facts": ["<exact figure or condition from a section>", ...],
 "decision": "answer|clarify|escalate|refuse",
 "calc": "<date lookup, or empty string>",
 "reply": "<message to the customer, containing every key fact>",
 "used_sections": ["<SECTION-ID>", ...]}"""


BASE_RULES = SUPPORT_RULES + JSON_FORMAT

NO_DOCS_NOTE = "\nYou have no access to company documents; use general e-commerce knowledge. used_sections must be []."

RAG_NOTE = """
Answer ONLY from the CONTEXT sections below. Each section has an ID, a status and an effective_from date.
- Prefer status=current. Use a superseded section only if the order/event date falls before the newer
  policy's effective_from date.
- When sources conflict, precedence is: policy > agent_guide > help > faq > marketing.
- If the CONTEXT does not cover the question, decision = "escalate": say you don't have that information
  and offer to connect a human colleague. Do not guess.
- List in used_sections ONLY the section IDs (or products.csv:SKU rows) you actually relied on."""


BASE_RULES = SUPPORT_RULES + JSON_FORMAT

NO_DOCS_NOTE = "\nYou have no access to company documents; use general e-commerce knowledge. used_sections must be []."

RAG_NOTE = """
Answer ONLY from the CONTEXT sections below. Each section has an ID, a status and an effective_from date.
- Prefer status=current. Use a superseded section only if the order/event date falls before the newer
  policy's effective_from date.
- When sources conflict, precedence is: policy > agent_guide > help > faq > marketing.
- If the CONTEXT does not cover the question, decision = "escalate": say you don't have that information
  and offer to connect a human colleague. Do not guess.
- List in used_sections ONLY the section IDs (or products.csv:SKU rows) you actually relied on."""

DECISIONS = {"answer", "clarify", "escalate", "refuse"}
NO_RETRY = ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError")

_last_call = [0.0]


def _throttle(messages):
    """Pace calls under a tokens-per-minute limit (LLM_TPM in .env; 0 = off)."""
    if not LLM_TPM:
        return
    est_tokens = sum(len(m["content"]) for m in messages) / 3.5 + 800   # prompt + reply estimate
    wait = _last_call[0] + 60 * est_tokens / LLM_TPM - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def chat(messages, temperature= 1, max_retries=5, json_mode=True):
    """One LLM call with retries. Returns the raw text, or raises after the last retry."""
    kw = dict(model=LLM_MODEL, messages=messages, temperature=temperature,top_p=0.95,parallel_tool_calls=True)
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    last_err = None
    for attempt in range(max_retries):
        _throttle(messages)
        try:
            r = client.chat.completions.create(**kw)
            return r.choices[0].message.content
        except Exception as e:
            last_err = e
            name = e.__class__.__name__
            if "response_format" in str(e) and "response_format" in kw:
                kw.pop("response_format"); continue          # provider has no JSON mode
            if name in NO_RETRY:
                raise                                        # retrying won't fix a bad key/model
            if "per day" in str(e):
                raise RuntimeError("Daily quota exhausted: " + str(e)[:200])
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 3 * 2 ** attempt
            print(f"  LLM error ({name}): retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def ask(question, context=None, history=None):
    """context=None -> baseline (no docs); context=str -> RAG.
    Always returns a dict with llm_ok (API call succeeded) and json_ok (model returned valid JSON)."""
    system = BASE_RULES + (RAG_NOTE + "\n\nCONTEXT:\n" + context if context is not None else NO_DOCS_NOTE)
    msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": question}]
    try:
        raw = chat(msgs)
    except Exception as e:
        return {"llm_ok": False, "json_ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "decision": None, "reply": "", "used_sections": []}

    out = parse_json(raw)
    if not isinstance(out, dict):
        return {"llm_ok": True, "json_ok": False, "error": None,
                "decision": "answer", "reply": raw or "", "used_sections": []}
    decision = str(out.get("decision", "")).lower().strip()
    return {"llm_ok": True, "json_ok": decision in DECISIONS, "error": None,
            "decision": decision if decision in DECISIONS else "answer",
            "reply": str(out.get("reply", "")),
            "used_sections": list(out.get("used_sections") or [])}


def test_connection(question="say hello"):
    """Call manually: from llm import test_connection; test_connection()"""
    print(LLM_BASE_URL, LLM_MODEL, (LLM_API_KEY or "")[:5] + "...", len(LLM_API_KEY or ""))
    print(ask(question))
