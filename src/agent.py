"""Step 3: the conversation agent (LLM + tools + memory) and its evaluator. Library only.

THE AGENT
  Agent().turn(text)      one customer message -> loop: LLM -> tools -> LLM ... -> final_answer.
                          Returns decision, reply, used_sections, retrieved, tool_calls, ticket_ids.
  Tools the LLM can call  search_kb (Step 2 retrieval) | lookup_order (identity-checked) |
                          create_ticket (handover) | date_calc (business / calendar days) | final_answer
  Memory                  self.history (all turns), self.orders (verified orders), self.tickets.
  Safety in code          max 2 verification attempts, pincode never shown to the LLM, one ticket per issue,
                          dates computed in Python ({date1} placeholders), AI intro on turn 1.
  Voice options           Agent(prefetch=True, search_k=8, style_note=..., one_ticket=True), used by voice.py.

THE EVALUATOR (practice/conversations.csv)
  run_conversations()     run all 23 conversations turn by turn (cached in outputs/cache_conv_<RUN_TAG>_*.json)
  score_conversations()   score every turn: decision, facts, violations, expected tool calls, retrieval
  conv_summary(df)        headline metrics

  from agent import Agent, run_conversations, score_conversations, conv_summary
"""
import json
import re
import time
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from config import DATASET_DIR, PRACTICE_DIR, OUT_DIR, LLM_MODEL, LLM_TPM, TODAY
from llm import client, SUPPORT_RULES, DECISIONS, NO_RETRY, parse_json
from rag_pipeline import retrieve, format_context, chunk_kb, HOLIDAYS
from metrics import score, split, retrieval_metrics

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("zentromart_tools", DATASET_DIR / "tools" / "tools.py")
_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tools)                     # load tools.py by path: works from any folder
_lookup_order, _create_ticket = _tools.lookup_order, _tools.create_ticket

MAX_VERIFY_ATTEMPTS = 2          # first try + one retry
SEARCH_K = 16                    # sections per search_kb call (the agent writes English queries; 26 hit rate limits)
KEEP_SEARCH_TURNS = 1            # full search results kept only for the current turn (context window)

# Standing rules: agent guides (escalation, verification, style), KB meta rules (precedence, business days) and
# current policy sections that say the agent must hand over. Found by doc_type / text at runtime, not by ID.
_HANDOVER_RE = re.compile(r"hand over|human|Priority Resolution", re.I)
STANDING = [c for c in {c["id"]: c for c in chunk_kb()
                        if c["file"] != "products.csv" and c["status"] == "current"
                        and (c["doc_type"] in ("agent_guide", "meta")
                             or (c["doc_type"] == "policy" and _HANDOVER_RE.search(c["text"])))}.values()]
STANDING_IDS = [c["id"] for c in STANDING]


TOOL_RULES = """

You have tools. Use them, and never say you did something a tool did not do.

TOOLS
- search_kb(query): search company documents and the product catalog. Call it before answering any policy,
  timeline, fee, warranty or product question, also on follow-up turns. Write the query in English using
  document words (e.g. "refund timeline UPI", "reverse pickup fee change of mind", "cancel shipped order")
  and include the product name or SKU if known. Search again if the first results miss the topic.
  Answer ONLY from what search_kb returned. Prefer status=current; use a superseded section only if the
  order/delivery date is before the newer policy's effective_from.
  Source priority when sources conflict: policy > agent_guide > help > faq > marketing.
  If the results do not cover the question at all, say you don't have that information and offer a human.
- lookup_order(order_id, verification_value): the first time an order ID appears, call it with ONLY
  order_id. Then ask for EITHER the last 4 digits of the registered phone OR the delivery pincode.
  EVERY time the customer gives a phone suffix or pincode (first try, retry, or in the same message as the
  order ID), you MUST call lookup_order with it before replying. Never judge a value yourself and never
  skip the call on a retry. Never share order details before verified=true; never say the expected value.
  Once verified, reuse the order details in this conversation; do not ask again.
- create_ticket(summary, order_id, reason): hand over to a human. Say a ticket was created only if the tool
  returned created=true, and give the ticket ID. Summary: 1-2 sentences, what the customer wants and why a
  human is needed; no phone digits or pincodes. Create ONE ticket per issue: if a ticket for this issue
  already exists in this conversation, do not call it again; just give the existing ID.
- date_calc(start_date, days, kind): business = refunds and delivery; calendar = return windows.
- Dates in the reply: never write a computed date yourself. Put {"start","days","kind"} in
  final_answer.dates (for a range "5 to 7" use the LAST number) and write {date1}, {date2} in the reply.
  Start from the order-record date (delivered_on for returns, *_initiated_on for refunds).

DECISION TABLE (check rows in this order; the first row that matches wins)

| # | Condition                                                                                     | Decision  |
|---|-----------------------------------------------------------------------------------------------|-----------|
| 1 | Verification turn, details match                                                              | answer    |
| 2 | Verification turn, first mismatch                                                             | clarify   |
| 3 | Verification turn, second mismatch                                                            | refuse    |
| 4 | Verified order AND a HARD TRIGGER applies (see B0)                                            | escalate  |
| 5 | Customer asks you to DO something the policy forbids (e.g. cancel a shipped order)            | refuse    |
| 6 | Information / status / eligibility question, or an action the customer is allowed to perform  | answer    |
| 7 | Required information is missing AND no hard trigger applies                                   | clarify   |

STEP 1 - CLASSIFY THE TURN (do this first, in order; these rules override everything below)

A) VERIFICATION TURN: the customer's message only supplies a phone suffix/pincode (optionally with the order
   ID) and asks nothing else, or you just asked for it.
   After lookup_order:
   - verified=true -> decision="answer". Reply = confirm they are verified + the order status and its key
     date in ONE sentence + ask how you can help. In this turn:
       * do NOT call create_ticket and do NOT mention escalation, a human, or a ticket;
       * do NOT mention items, prices, damage, refunds, return reasons or contact counts, even if the
         order record shows a high-value, damaged, overdue or repeat-contact case. Hand-over rules are
         applied in the NEXT turn when the customer sends any message.
   - verified=false, FIRST failure -> decision="clarify". The reply MUST say the details did not match
     (English: "did not match"; Hinglish: "match nahi hue"; Hindi: "मेल नहीं खाए") and ask them to try
     AGAIN (again / dobara / दोबारा) with the last 4 digits of the registered phone or the delivery pincode.
     Say one retry is left. Share no order details.
   - verified=false, SECOND failure -> decision="refuse". Call lookup_order first (mandatory), do NOT call
     create_ticket. Reply must: start with an apology (English "Sorry,"; Hinglish "Maaf kijiye,";
     Hindi "क्षमा करें,"), say you cannot share order details (English "cannot share"; Hinglish
     "share nahi kar sakta"; Hindi "साझा नहीं कर सकता"), tell them to contact support from their
     REGISTERED phone number (पंजीकृत / registered), and offer help with general policy questions.
     Do not promise a human callback.

B) ISSUE TURN: any message on a verified order that is NOT a verification turn.

   B0) HARD TRIGGERS - check these BEFORE choosing any decision. If any one applies, decision="escalate"
       and you must call create_ticket in THIS turn (unless a ticket for this issue already exists):
       T1. contacts_last_7_days >= 2 in the order record. This applies to ANY non-verification message,
           even a vague one ("I need help", "problem with my order", "hello"). Do NOT ask what the issue
           is, do NOT use "clarify". Call create_ticket(reason="repeat_contact"). Reply: acknowledge they
           have contacted us before about this order, say a human colleague will take over, give the
           ticket ID.
       T2. A refund is MORE than 3 business days past its maximum timeline. Compute with date_calc from
           *_initiated_on (timeline end + 3 business days). A refund still inside its timeline is NOT
           overdue. See OVERDUE REFUND REPLY below.
       T3. Customer reports a damaged/defective/wrong item worth more than ₹10,000, or asks for a
           replacement/refund approval for it.
       T4. Customer asks for a human, reports an order/payment they did not make, makes a legal threat, or
           raises a safety issue.
       Exception: a pure status question on a return that was already picked up/initiated is NOT an
       escalation, even for an item over ₹10,000, unless the refund is overdue (T2) or T1 applies.
       The order record may hold high-value or wrong-item data from an earlier return. That data alone is
       NOT a trigger; only what the customer is asking NOW counts (T1 and T2 are judged from the record).
       When a hard trigger applies, escalate, never refuse and never clarify.

   B1) Identify what the customer is asking:
       (a) STATUS / TIMELINE question about something in progress
       (b) REPORTING A PROBLEM or asking for a decision
       (c) REQUEST FOR AN ACTION (cancel, voucher, compensation, exception)
       (d) ELIGIBILITY question ("can I return this?")

   B2) Choose the decision (only if no hard trigger applied):

   1. "escalate": already decided by B0. After create_ticket returns created=true, say a human colleague
      will take over and give the ticket ID.

   2. "refuse": the customer asks you to DO something the policy does NOT allow.
      The availability of an alternative does NOT turn a refusal into an answer.
      MANDATORY SHIPPED-ORDER CANCELLATION RULE:
        If the customer requests cancellation AND order status is shipped, out_for_delivery or delivered,
        decision MUST be "refuse". Always. Offering alternatives (refuse delivery, return) does not
        change this.
      Reply order:
        1. Apology first (English "Sorry,"; Hinglish "Maaf kijiye,"; Hindi "क्षमा करें,").
        2. Say clearly that cancellation is not possible.
        3. Reason: the order has already shipped (or is out for delivery / delivered).
        4. Allowed alternatives from the policy (refuse the parcel at delivery; return after delivery),
           with exact windows, fees, conditions and refund timelines. Use search_kb and the RETURN
           ELIGIBILITY checklist for the numbers.
      Examples:
        - Shipped order + "cancel it" = refuse.
        - Out-for-delivery + "cancel" = refuse.
        - Delivered + "cancel" = refuse.
        - "How do I refuse delivery?" = answer (information question).
        - "Can I return this delivered item?" = answer (eligibility question).

   3. "answer": everything the policy allows or explains:
      - Informational questions, including "can I return this?" when the answer is no. This does NOT
        include requests to perform an action the policy forbids (that is "refuse").
      - Status/timeline questions.
      - Actions the customer is allowed to perform.
      - Cancellation of placed / confirmed / packed orders that have NOT shipped: say cancellation is free
        of charge from My Orders, state the refund timeline, do NOT apologise or refuse.

   4. "clarify": only if NO hard trigger (B0) applies AND you lack information needed to answer (e.g. which
      order/item). Never use clarify when contacts_last_7_days >= 2.

   OVERDUE REFUND REPLY (when T2 applies). The reply MUST contain ALL of:
     (1) refund amount + payment method,
     (2) the initiation date (use a date placeholder from dates),
     (3) the policy timeline and the {date1} by which it was due,
     (4) the RRN copied EXACTLY from the order record (refund RRN field), digits unchanged,
     (5) the ticket ID returned by create_ticket,
     (6) a statement that a human colleague will take over.
   Put the RRN and the ticket ID in key_facts as well. Never omit the RRN. If the order record has no RRN,
   say so explicitly instead of inventing one.

   RETURN ELIGIBILITY / RETURN WINDOW answers (mandatory checklist, in this order):
     1. Search "return window by category" AND search "festive sale extended return window" (or the
        sale/promotion name) AND search "reverse pickup fee".
     2. Compare the order's PLACED date (ordered_on / order_date) with every sale or promotion period you
        found. If the order falls inside a sale period that extends the return window, use the EXTENDED
        window, not the standard category window. Mention that the sale extension applies.
     3. Start date for date_calc = delivered_on; kind="calendar"; days = the window that applies.
     4. State: window in days, the deadline as {date1}, and the reverse pickup fee: if the reason is change
        of mind and the customer is not Plus, state the fee (₹49 if the document says so); if Plus or
        the item is damaged/wrong, say there is no pickup fee.

   REFUND STATUS answers (mandatory content, when the refund is NOT overdue):
     - the amount/payment method if known from the record.
     - State the complete refund process:
         1. The item must pass the quality check.
         2. The refund starts within N business days after pickup.
         3. State the payment-method timeline (e.g. UPI 1 to 3 business days).
     - TERMINOLOGY: the phrase MUST appear exactly as "quality check" or "गुणवत्ता जाँच".
       Never use "क्वालिटी चेक", "क्वालिटी जांच" or other transliterations, even in a fully Hindi reply.
       Hindi example: "वस्तु की गुणवत्ता जाँच के बाद refund 2 business days के अंदर शुरू होगा।"
       Hinglish example: "Quality check pass hone ke baad refund 2 business days ke andar initiate hoga."
     - Say Plus status changes the timeline only if is_plus is true.
     - If the refund is overdue by T2, follow OVERDUE REFUND REPLY instead.

STEP 2 - WRITING THE REPLY
- First reply of the conversation: say you are ZentroMart's AI assistant.
- Reply in the customer's language (Hindi script, Hinglish in Latin script, or English). Match their script.
- Use the policy's own terms for key concepts, in the document's wording, even inside Hindi/Hinglish:
  write "quality check" (or गुणवत्ता जाँच), "reverse pickup fee", "business days", "RRN", "Plus", "ticket ID".
  Do not transliterate them into Devanagari (not "क्वालिटी चेक").
- Every figure the answer depends on (days, ₹, fee, timeline, date placeholder, ticket ID, RRN) must appear
  in the reply and in key_facts. Include the process condition the policy attaches to a refund
  (e.g. "after the item passes the quality check").
- Answer only what was asked. Do not mention tickets or human hand-over unless decision="escalate".
- Resolve follow-ups ("how long will that take?") against the same order and item; do not re-ask.
- With a verified order, combine the record (return_category, delivered_on, payment method, is_plus,
  refund dates, RRN) with the policy.
- Write each reply in ONE script. Do not mix Latin letters inside Hindi words (e.g. never "वrong").
  Brand/policy terms (ZentroMart, Plus, RRN, ticket ID, quality check, My Orders) may stay in Latin script.

FINISHING
Before calling final_answer, run this check:
  1. If your reply tells the customer you cannot do what they asked (cancel, etc.), decision MUST be
     "refuse", not "answer".
  2. If contacts_last_7_days >= 2 and this is not a verification turn, decision MUST be "escalate" and
     create_ticket must have returned created=true.
  3. If decision="escalate" for an overdue refund, the RRN and ticket ID are in the reply and in key_facts.
  4. Every number, date placeholder and ID in the reply is also in key_facts.
Call final_answer(key_facts, decision, reply, used_sections, dates) exactly once. No JSON as plain text,
no tool named "json". used_sections = only the section IDs (or products.csv:SKU rows) whose facts you
actually used in the reply; include a STANDING RULES ID only if it drove the decision. Do not list
sections you merely retrieved."""




SYSTEM = (SUPPORT_RULES + TOOL_RULES + "\n\nSTANDING RULES (agent guides and hand-over policies; always apply):\n"
          + format_context(STANDING))

TOOLS = [
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search ZentroMart policies, agent guides, help articles, FAQs and the product "
                       "catalog. Returns sections with their ID, status and effective_from date.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "English query using document vocabulary; "
                                                       "include product name or SKU when known."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Identity-checked order lookup. Without verification_value it returns only a "
                       "verification prompt. With a matching value it returns the order details.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "8-digit order ID"},
            "verification_value": {"type": "string",
                                   "description": "Last 4 digits of the registered phone OR the 6-digit "
                                                  "delivery pincode, as the customer gave it. Omit on "
                                                  "the first lookup."}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Hand the case to a human colleague. Returns created and ticket_id.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "1-2 sentences for the human colleague; "
                                                         "no phone digits or pincodes."},
            "order_id": {"type": "string"},
            "reason": {"type": "string", "description": "short code, e.g. high_value_damage, "
                                                        "policy_exception, compensation_request, "
                                                        "unauthorized_order, verification_failed, "
                                                        "not_covered"}},
            "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "date_calc",
        "description": "Add days to a date. business = Mon-Sat excluding holidays; calendar = every day.",
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "days": {"type": "integer"},
            "kind": {"type": "string", "enum": ["business", "calendar"]}},
            "required": ["start_date", "days", "kind"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Send the final reply to the customer. Call this exactly once to end the turn.",
        "parameters": {"type": "object", "properties": {
            "key_facts": {"type": "array", "items": {"type": "string"},
                          "description": "Exact figures and conditions (days, months, ₹, dates, ticket ID) "
                                         "from sources and tools that the reply must contain."},
            "decision": {"type": "string", "enum": ["answer", "clarify", "escalate", "refuse"]},
            "reply": {"type": "string", "description": "Message to the customer, in their language, "
                                                       "containing every key fact."},
            "used_sections": {"type": "array", "items": {"type": "string"}},
            "dates": {"type": "array", "description": "One item per {dateN} placeholder in the reply.",
                      "items": {"type": "object", "properties": {
                          "start": {"type": "string", "description": "YYYY-MM-DD"},
                          "days": {"type": "integer"},
                          "kind": {"type": "string", "enum": ["business", "calendar"]}},
                          "required": ["start", "days", "kind"]}}},
            "required": ["key_facts", "decision", "reply", "used_sections"]}}},
]
FINAL_CHOICE = {"type": "function", "function": {"name": "final_answer"}}

# --------------------------------------------------------------------------- tool helpers


def date_calc(start_date, days, kind="business"):
    d, n = date.fromisoformat(str(start_date)[:10]), int(days)
    if kind == "calendar":
        end = d + timedelta(days=n)
    else:
        end, left = d, n
        while left > 0:                                  # day 1 = next business day
            end += timedelta(days=1)
            if end.weekday() != 6 and end.isoformat() not in HOLIDAYS:
                left -= 1
    return {"start_date": d.isoformat(), "days": n, "kind": kind, "result_date": end.isoformat(),
            "weekday": end.strftime("%A"), "already_passed": end.isoformat() < TODAY}


def business_days_between(start, end=TODAY):
    d, e, n = date.fromisoformat(str(start)[:10]), date.fromisoformat(end), 0
    while d < e:
        d += timedelta(days=1)
        if d.weekday() != 6 and d.isoformat() not in HOLIDAYS:
            n += 1
    return n


MONTHS_HI = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


def fill_dates(reply, specs):
    """Replace {date1}, {date2}, ... with dates computed in code from the model's specs."""
    deva = bool(re.search(r"[ऀ-ॿ]", reply))
    for i, s in enumerate(specs or [], 1):
        try:
            d = date.fromisoformat(date_calc(s["start"], s["days"], s.get("kind", "business"))["result_date"])
        except Exception:
            continue
        reply = reply.replace(f"{{date{i}}}", f"{d.day} {MONTHS_HI[d.month - 1]} {d.year}" if deva
                              else f"{d.day} {d:%B %Y}")
    return re.sub(r"\{date\d+\}", "", reply)


def parse_first_json(text):
    """First JSON object in text (the model sometimes writes the same object twice)."""
    text = text or ""
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i = text.find("{", i + 1)
    return None


def looks_broken(reply):
    """Degenerate generations: empty, or long runs of ellipses / punctuation."""
    r = (reply or "").strip()
    return len(r) < 5 or r.count("…") + r.count("...") >= 5 or len(re.findall(r"\w", r)) < 0.4 * len(r)


AI_INTRO = {"deva": "मैं ZentroMart का AI सहायक हूँ। ", "en": "I am ZentroMart's AI assistant. "}


def _digits(x):
    return re.sub(r"\D", "", str(x or ""))


def _compact(d):
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def _dumps(x):
    return json.dumps(x, ensure_ascii=False, default=str)


def _args(raw):
    try:
        a = json.loads(raw or "{}")
        return a if isinstance(a, dict) else {}
    except Exception:
        return {}

# --------------------------------------------------------------------------- LLM call with tools


_last_call = [0.0]


def _pace(messages):
    if not LLM_TPM:
        return
    chars = sum(len(str(m.get("content") or "")) + len(_dumps(m.get("tool_calls") or "")) for m in messages)
    wait = _last_call[0] + 60 * (chars / 3.5 + 800) / LLM_TPM - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _recover_final(e):
    """Groq rejects a call to a non-existent tool (e.g. 'json') but returns the attempted call in
    failed_generation. If it carries a reply, use it as the final answer instead of failing."""
    body = getattr(e, "body", None) or {}
    if isinstance(body, dict):
        body = body.get("error", body)
    gen = body.get("failed_generation") if isinstance(body, dict) else None
    data = parse_json(gen or "")
    if isinstance(data, dict):
        args = data.get("arguments", data)
        if isinstance(args, str):
            args = parse_json(args)
        if isinstance(args, dict) and "reply" in args:
            return SimpleNamespace(content=_dumps(args), tool_calls=None)
    return None


REASONING = None        # None = model default. "low" / "off" = less hidden thinking (faster); set by call_handsfree.py --fast
_REASONING_UNSUPPORTED = set()                        # models that rejected the setting (then sent without it)


def _reasoning_kwargs(model):
    """Provider-specific switch for hidden reasoning. gpt-oss: reasoning_effort; Nemotron/Qwen3: thinking off."""
    if not REASONING or model in _REASONING_UNSUPPORTED:
        return {}
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": "low"}
    if "nemotron" in m or "qwen3" in m:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def _complete(messages, tool_choice="auto", max_retries=5, temperature=0.0):
    kw = dict(model=LLM_MODEL, messages=messages, temperature=temperature, tools=TOOLS, tool_choice=tool_choice)
    kw.update(_reasoning_kwargs(LLM_MODEL))
    last_err = None
    for attempt in range(max_retries):
        _pace(messages)
        try:
            return client.chat.completions.create(**kw).choices[0].message
        except Exception as e:
            last_err, name, s = e, e.__class__.__name__, str(e)
            if ("reasoning_effort" in kw or "extra_body" in kw) and name == "BadRequestError":
                _REASONING_UNSUPPORTED.add(LLM_MODEL)                # model does not accept it: drop and retry
                kw.pop("reasoning_effort", None); kw.pop("extra_body", None)
                continue
            if "tool_use_failed" in s or "Failed to call a function" in s:
                rec = _recover_final(e)
                if rec is not None:
                    return rec
                kw["temperature"] = 0.4                       # resample differently, not the same output
                time.sleep(1); continue
            if name in NO_RETRY:
                raise
            if "per day" in s:
                raise RuntimeError("Daily quota exhausted: " + s[:200])
            m = re.search(r"try again in ([\d.]+)s", s)
            wait = float(m.group(1)) + 1 if m else 3 * 2 ** attempt
            print(f"  LLM error ({name}): retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")

# --------------------------------------------------------------------------- agent


class Agent:
    """One conversation. Verified order state lives only in this object."""

    def __init__(self, max_steps=6, prefetch=False, search_k=None, style_note="", one_ticket=False):
        """prefetch=True (voice mode): search the KB in code before the first LLM call, so the LLM can usually
        answer in one call instead of two. search_k: sections per search (default SEARCH_K).
        style_note: extra rules appended to the system prompt (e.g. short replies for a phone call).
        one_ticket: at most one ticket per conversation; later create_ticket calls return the existing one."""
        self.prefetch = prefetch
        self.system = SYSTEM + style_note
        self.one_ticket = one_ticket
        self.search_k = search_k or SEARCH_K
        self.history = []          # user / assistant / tool messages across turns
        self.orders = {}           # order_id -> verified lookup result
        self.failed = {}           # order_id -> failed verification attempts
        self.tickets = {}          # (order_id, reason) -> ticket result, so a handover is not duplicated
        self.search_msgs = []      # (turn_no, history index, query, ids) of search_kb tool messages
        self.turn_no = 0
        self.max_steps = max_steps

    # ---- context window: old search results are replaced by a one-line stub
    def _compact_history(self):
        for t, idx, query, ids in self.search_msgs:
            if t <= self.turn_no - KEEP_SEARCH_TURNS and not self.history[idx]["content"].startswith("[earlier"):
                self.history[idx]["content"] = (f"[earlier search_kb('{query}') returned {', '.join(ids[:8])}; "
                                                f"details omitted, search again if needed]")

    # ---- tools
    def _search(self, a):
        hits = retrieve(str(a.get("query", "")), self.search_k)
        ids = list(dict.fromkeys(h["id"] for h in hits))
        return format_context(hits), {"name": "search_kb", "query": a.get("query")}, ids

    def _lookup(self, a):
        oid = _digits(a.get("order_id"))
        vv = _digits(a.get("verification_value")) or None
        log = {"name": "lookup_order", "order_id": oid, "verification": bool(vv)}   # never log the value
        if not vv and oid in self.orders:
            res = self.orders[oid]
        elif vv and self.failed.get(oid, 0) >= MAX_VERIFY_ATTEMPTS:
            res = {"verified": False, "error": "verification_locked",
                   "note": "Two failed attempts. Share no order details; offer a human colleague."}
        else:
            res = dict(_lookup_order(oid, verification_value=vv))
            res.pop("delivery_pincode", None)                    # verification secret: never shown to the LLM
            if vv:
                if res.get("verified"):
                    res["today"] = TODAY                         # facts computed in code, not by the LLM
                    for k in [k for k in res if k.endswith("_initiated_on") and res[k]]:
                        res[k.replace("_initiated_on", "_business_days_since_initiated")] = \
                            business_days_between(res[k])
                    self.orders[oid] = res
                else:
                    self.failed[oid] = self.failed.get(oid, 0) + 1
                    res["attempts_left"] = max(0, MAX_VERIFY_ATTEMPTS - self.failed[oid])
        log["verified"] = bool(res.get("verified"))
        return _dumps(_compact(res)), log, []

    def _ticket(self, a):
        oid = _digits(a.get("order_id")) or None
        key = (oid, a.get("reason"))
        if self.one_ticket and self.tickets and key not in self.tickets:   # one handover per call is enough
            key = next(iter(self.tickets))
        if key in self.tickets:                                  # same handover already created
            res = self.tickets[key] | {"note": "ticket already exists for this issue"}
        else:
            res = dict(_create_ticket(summary=str(a.get("summary", "")), order_id=oid, reason=a.get("reason")))
            if res.get("created"):
                self.tickets[key] = res
        ok = bool(res.get("created"))
        log = {"name": "create_ticket", "order_id": oid, "reason": a.get("reason"),
               "created": ok, "ticket_id": res.get("ticket_id") if ok else None}
        return _dumps(res), log, []

    def _date(self, a):
        res = date_calc(a.get("start_date"), a.get("days", 0), a.get("kind", "business"))
        return _dumps(res), {"name": "date_calc", **res}, []

    def _run_tool(self, name, a):
        fn = {"search_kb": self._search, "lookup_order": self._lookup,
              "create_ticket": self._ticket, "date_calc": self._date}.get(name)
        if fn is None:
            return _dumps({"error": f"unknown tool {name}"}), {"name": name, "error": "unknown"}, []
        try:
            return fn(a)
        except Exception as e:
            return _dumps({"error": f"{type(e).__name__}: {e}"}), {"name": name, "error": str(e)[:200]}, []

    # ---- one customer message
    def turn(self, text):
        self.turn_no += 1
        self._compact_history()
        self.history.append({"role": "user", "content": text})
        calls, retrieved, raw = [], [], ""
        if self.prefetch and len(re.findall(r"[^\W\d_]{2,}", text)) >= 3:   # a real question, not just an ID/digits
            tc_id = f"prefetch_{self.turn_no}"                              # looks like a search_kb call the LLM made
            content, log, ids = self._search({"query": text})
            self.history.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": tc_id, "type": "function",
                 "function": {"name": "search_kb", "arguments": json.dumps({"query": text}, ensure_ascii=False)}}]})
            self.history.append({"role": "tool", "tool_call_id": tc_id, "content": content})
            self.search_msgs.append((self.turn_no, len(self.history) - 1, text, ids))
            calls.append(log)
            retrieved += ids
        try:
            for step in range(self.max_steps + 1):
                final_only = step == self.max_steps
                msg = _complete([{"role": "system", "content": self.system}] + self.history,
                                tool_choice=FINAL_CHOICE if final_only else "auto")
                fa = next((tc for tc in (msg.tool_calls or []) if tc.function.name == "final_answer"), None)
                if fa is not None:
                    raw = fa.function.arguments or ""
                    break
                if msg.tool_calls and not final_only:
                    self.history.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        content, log, ids = self._run_tool(tc.function.name, _args(tc.function.arguments))
                        calls.append(log)
                        retrieved += ids
                        self.history.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                        if tc.function.name == "search_kb":
                            self.search_msgs.append((self.turn_no, len(self.history) - 1, log.get("query"), ids))
                    continue
                raw = msg.content or ""
                break
        except Exception as e:
            return {"llm_ok": False, "json_ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "decision": None, "reply": "", "used_sections": [], "retrieved": [],
                    "tool_calls": calls, "ticket_ids": []}

        out = parse_first_json(raw)
        if not isinstance(out, dict) or looks_broken(str(out.get("reply", ""))):
            try:                                                 # one retry, forced to call final_answer
                msg = _complete([{"role": "system", "content": self.system}] + self.history, tool_choice=FINAL_CHOICE,
                                temperature=0.4)
                fa = next((tc for tc in (msg.tool_calls or []) if tc.function.name == "final_answer"), None)
                retry = parse_first_json(fa.function.arguments if fa else msg.content)
                if isinstance(retry, dict) and not looks_broken(str(retry.get("reply", ""))):
                    out = retry
            except Exception:
                pass
        if isinstance(out, dict):
            decision = str(out.get("decision", "")).lower().strip()
            json_ok, reply = decision in DECISIONS, fill_dates(str(out.get("reply", "")), out.get("dates"))
            used, facts = list(out.get("used_sections") or []), list(out.get("key_facts") or [])
        else:
            decision, json_ok, reply, used, facts = "answer", False, raw, [], []
        if self.turn_no == 1 and not re.search(r"\bAI\b|एआई", reply):     # STY-1: introduce as AI assistant
            reply = AI_INTRO["deva" if re.search(r"[ऀ-ॿ]", reply) else "en"] + reply
        # store only the reply text: a JSON string in history teaches the model to answer in raw JSON
        self.history.append({"role": "assistant", "content": reply})
        return {"llm_ok": True, "json_ok": json_ok, "error": None,
                "decision": decision if decision in DECISIONS else "answer",
                "reply": reply, "used_sections": used, "key_facts": facts,
                "retrieved": list(dict.fromkeys(retrieved + STANDING_IDS)),
                "tool_calls": calls,
                "ticket_ids": [c["ticket_id"] for c in calls if c.get("ticket_id")]}

# --------------------------------------------------------------------------- evaluation on conversations.csv


def load_conversations():
    return pd.read_csv(PRACTICE_DIR / "conversations.csv", dtype=str, keep_default_na=False)


RUN_TAG = "v4"                # change (e.g. "v3") after editing the prompt/tools to keep old results


def conv_cache_path():
    return OUT_DIR / f"cache_conv_{RUN_TAG}_{LLM_MODEL.replace('/', '_')}.json"


def _load_cache():
    p = conv_cache_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_cache(cache):
    conv_cache_path().write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def run_conversations(limit=None, ids=None):
    """Run each conversation turn by turn with a fresh Agent. Successful conversations are cached;
    a conversation with any failed LLM call is rerun from turn 1 next time."""
    conv = load_conversations()
    cids = sorted(conv.conversation_id.unique())
    if ids:
        cids = [c for c in cids if c in set(ids)]
    if limit:
        cids = cids[:limit]
    cache = _load_cache()
    todo = [c for c in cids if cache.get(c, {}).get("llm_status") != "success"]
    print(f"conversations: {len(cids) - len(todo)} cached, {len(todo)} to run")
    for cid in todo:
        g = conv[conv.conversation_id == cid].sort_values("turn", key=lambda s: s.astype(int))
        agent, turns, ok = Agent(), [], True
        for _, row in g.iterrows():
            out = agent.turn(row.customer_message)
            turns.append({"turn": row.turn, **out})
            print(f"  {cid} t{row.turn}: {out['decision'] or out['error']}  "
                  f"tools={[c['name'] for c in out['tool_calls'] if c['name'] != 'search_kb']}")
            if not out["llm_ok"]:
                ok = False; break
        err = str(turns[-1].get("error")) if not ok else ""
        if "Daily quota" in err:
            print("  Groq daily quota reached: stopping. Rerun tomorrow; finished conversations stay cached.")
            break
        if any(x in err for x in ("NotFoundError", "AuthenticationError", "PermissionDeniedError")):
            print(f"  Stopping: model or API key problem, check .env (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).\n  {err[:200]}")
            break
        cache[cid] = {"llm_status": "success" if ok else "failed", "turns": turns}
        _save_cache(cache)
    n_failed = sum(1 for c in cids if cache.get(c, {}).get("llm_status") != "success")
    print(f"conversations: done — {n_failed} still failed" + (" (rerun to retry them)" if n_failed else ""))
    return cache


CALL_RE = re.compile(r"(\w+)\s*\(([^)]*)\)")


def parse_expected_calls(cell):
    out = []
    for name, args in CALL_RE.findall(cell or ""):
        kv = dict(re.findall(r"(\w+)\s*=\s*['\"]?([^,'\"]+)", args))
        out.append({"name": name, "order_id": _digits(kv.get("order_id")) or None,
                    "verification": "verification_value" in kv})
    return out


def tool_hits(expected, actual):
    remaining, hits = [c for c in actual if c["name"] != "search_kb"], 0
    for e in expected:
        for a in remaining:
            if (a["name"] == e["name"]
                    and (e["order_id"] is None or a.get("order_id") == e["order_id"])
                    and (e["name"] != "lookup_order" or a.get("verification") == e["verification"])):
                remaining.remove(a); hits += 1; break
    return hits


def score_conversations():
    """Score cached turns. No LLM calls."""
    conv, cache, rows = load_conversations(), _load_cache(), []
    for _, row in conv.iterrows():
        c = cache.get(row.conversation_id)
        if not c or c.get("llm_status") != "success":
            continue
        out = next((t for t in c["turns"] if str(t["turn"]) == str(row.turn)), None)
        if out is None:
            continue
        target = SimpleNamespace(must_include=row.must_include, must_not_include=row.must_not_include,
                                 expected_decision=row.expected_decision,
                                 acceptable_decisions=row.expected_decision)
        s = score(target, out)
        exp = parse_expected_calls(row.expected_tool_calls)
        hits = tool_hits(exp, out["tool_calls"])
        ticket_made = any(t["name"] == "create_ticket" and t.get("created") for t in out["tool_calls"])
        gold = split(row.gold_evidence)
        rows.append({"conversation_id": row.conversation_id, "turn": int(row.turn), "scenario": row.scenario,
                     "language": row.language, "expected": row.expected_decision, "decision": out["decision"],
                     "expected_tools": row.expected_tool_calls,
                     "actual_tools": [t for t in out["tool_calls"] if t["name"] != "search_kb"],
                     "reply": out["reply"], "gold": gold, "retrieved": out.get("retrieved", [])}
                    | s
                    | {"tools_expected": len(exp), "tools_hit": hits, "tools_ok": hits == len(exp),
                       "unexpected_ticket": ticket_made and not any(e["name"] == "create_ticket" for e in exp),
                       "turn_correct": s["correct"] and hits == len(exp)}
                    | (retrieval_metrics(out.get("retrieved", []), gold, out.get("used_sections")) if gold else {}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"results_conversations_{RUN_TAG}.csv", index=False)
    return df


def conv_summary(df):
    if df.empty:
        print("Nothing scored yet: no conversation finished. Run run_conversations() again.")
        return {"turns scored": 0}
    by_conv = df.groupby("conversation_id").turn_correct.all()
    s = {"turns scored": len(df),
         "turns correct (reply + tools)": f"{int(df.turn_correct.sum())} / {len(df)} ({df.turn_correct.mean():.1%})",
         "reply correct (ignoring tools)": f"{df.correct.mean():.1%}",
         "decision accuracy": f"{df.decision_ok.mean():.1%}",
         "avg fact recall": f"{df.fact_recall.mean():.1%}",
         "violations": int(df.violation.sum()),
         "expected tool calls made": f"{int(df.tools_hit.sum())} / {int(df.tools_expected.sum())}",
         "unexpected tickets": int(df.unexpected_ticket.sum()),
         "conversations fully correct": f"{int(by_conv.sum())} / {len(by_conv)}"}
    for col in ("recall@k", "hit@k", "mrr", "cite_precision"):
        if col in df:
            s[col] = round(float(df[col].mean()), 3)
    return s
