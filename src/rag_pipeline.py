"""Step 2 RAG pipeline (the version that scored 50/52). Library only: importing it runs nothing.

  chunk_kb()            kb/*.md split by [SECTION-ID] + one chunk per product row
  retrieve(question)    hybrid search: dense (multilingual-e5) + BM25, fused with reciprocal rank fusion
  format_context(hits)  the retrieved sections as text for the prompt
  make_guide(question)  a date table (business / calendar days) + answer rules put in front of the context
  rag_answer(question)  retrieve -> guide + context -> LLM -> answer dict (decision, reply, used_sections...)

The vector index is built on the first retrieve() and saved to outputs/vector_index.pkl
(delete that file after changing the chunking or EMBED_MODEL).
"""
import pickle
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import KB_DIR, DATA_DIR, OUT_DIR, EMBED_MODEL, TOP_K, TODAY
from llm import ask

SECTION_RE = re.compile(r"^#{1,6}\s*\[([A-Za-z0-9_.-]+)\]\s*(.*)$", re.M)
MAX_CHARS = 1500
PRECEDENCE = {"policy": 0, "agent_guide": 1, "help": 2, "faq": 3, "marketing": 4}
INDEX_FILE = OUT_DIR / "vector_index.pkl"
HOLIDAYS = set(pd.read_csv(DATA_DIR / "holidays.csv", dtype=str, keep_default_na=False)["date"])

# ------------------------------------------------------------------ chunking

def parse_header(text):
    head = text.split("\n## ", 1)[0]
    meta = {}
    for line in head.splitlines():
        m = re.match(r"^\s*[-*]?\s*\**([a-z_]+)\**\s*:\s*(.+?)\s*$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip("`\"' ")
    return meta


def chunk_kb():
    chunks = []
    for f in sorted(KB_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        meta = parse_header(text)
        heads = list(SECTION_RE.finditer(text))
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            parts, cur = [], ""
            for p in text[m.end():end].strip().split("\n\n"):     # split on paragraphs
                if len(cur) + len(p) > MAX_CHARS and cur:
                    parts.append(cur); cur = ""
                cur += p + "\n\n"
            parts.append(cur)
            for j, part in enumerate(parts):
                chunks.append({
                    "id": m.group(1), "part": j, "title": m.group(2).strip(), "text": part.strip(),
                    "file": f.name, "doc_title": meta.get("title", f.stem),
                    "doc_type": meta.get("doc_type", ""), "status": meta.get("status", "current"),
                    "effective_from": meta.get("effective_from", ""), "language": meta.get("language", "en"),
                })
    p = pd.read_csv(DATA_DIR / "products.csv", dtype=str, keep_default_na=False)
    for r in p.itertuples():
        chunks.append({
            "id": f"products.csv:{r.sku}", "part": 0, "title": r.title, "file": "products.csv",
            "text": (f"SKU {r.sku}: {r.title} by {r.brand}. Return category {r.return_category}. "
                     f"Price ₹{r.price_inr}. Fulfilled by {r.fulfilled_by}. Assured badge: {r.assured}. "
                     f"Warranty {r.warranty_months} months. {r.description}"),
            "doc_title": "Product catalog", "doc_type": "catalog", "status": "current",
            "effective_from": "", "language": "en",
        })
    return chunks


def passage(c):
    return f"{c['doc_title']} — [{c['id']}] {c['title']}\n{c['text']}"


def tok(t):
    return re.findall(r"[\wऀ-ॿ]+", t.lower())


def format_context(hits):
    return "\n\n".join(
        f"### [{h['id']}] {h['title']}\n(source: {h['file']} | type: {h['doc_type']} | "
        f"status: {h['status']} | effective_from: {h['effective_from'] or 'n/a'})\n{h['text']}"
        for h in hits)

# ------------------------------------------------------------------ index + retrieval

_state = {}


def _index():
    """Build (or load) the index once per kernel."""
    if not _state:
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        embedder = SentenceTransformer(EMBED_MODEL)
        if INDEX_FILE.exists():
            chunks, emb = pickle.loads(INDEX_FILE.read_bytes())
        else:
            chunks = chunk_kb()
            emb = embedder.encode(["passage: " + passage(c) for c in chunks],
                                  normalize_embeddings=True, batch_size=32, show_progress_bar=True)
            INDEX_FILE.write_bytes(pickle.dumps((chunks, emb)))
        _state.update(embedder=embedder, chunks=chunks, emb=emb,
                      bm25=BM25Okapi([tok(passage(c)) for c in chunks]))
    return _state


def _rrf(scores):
    """Reciprocal rank fusion score for every chunk (1 / (60 + rank))."""
    ranks = np.empty(len(scores))
    ranks[np.argsort(-np.asarray(scores), kind="stable")] = np.arange(len(scores))
    return 1.0 / (60 + ranks)


def retrieve(question, k=TOP_K, w_dense=2.0, w_bm25=1.0, max_catalog=2):
    """Top-k sections for a question. Dense search counts twice as much as BM25; one slot per section ID;
    at most max_catalog product rows, so policy sections are not crowded out by product variants."""
    s = _index()
    dense = s["emb"] @ s["embedder"].encode(["query: " + question], normalize_embeddings=True)[0]
    fused = w_dense * _rrf(dense) + w_bm25 * _rrf(s["bm25"].get_scores(tok(question)))
    for i, c in enumerate(s["chunks"]):                   # mild priors, not filters
        if c["status"] == "superseded":
            fused[i] *= 0.85
        fused[i] *= 1 - 0.03 * PRECEDENCE.get(c["doc_type"], 2)
    for sku in re.findall(r"[A-Z]{3}-\d{4}", question.upper()):   # exact SKU mentioned -> that product row
        for i, c in enumerate(s["chunks"]):
            if c["id"] == f"products.csv:{sku}":
                fused[i] += 1
    out, seen, n_cat = [], set(), 0
    for i in np.argsort(-fused, kind="stable"):
        c = s["chunks"][i]
        if c["id"] in seen:
            continue
        if c["file"] == "products.csv":
            if n_cat >= max_catalog:
                continue
            n_cat += 1
        seen.add(c["id"])
        out.append(c | {"score": round(float(fused[i]), 4)})
        if len(out) >= k:
            break
    return out

# ------------------------------------------------------------------ date table + answer rules

_EN = ["january", "february", "march", "april", "may", "june", "july", "august",
       "september", "october", "november", "december"]
MONTHS = {m: i + 1 for i, m in enumerate(_EN)} | {m[:3]: i + 1 for i, m in enumerate(_EN)} | {"sept": 9}
for _i, _names in enumerate([["जनवरी"], ["फ़रवरी", "फरवरी"], ["मार्च"], ["अप्रैल"], ["मई"], ["जून"], ["जुलाई"],
                             ["अगस्त"], ["सितंबर", "सितम्बर"], ["अक्टूबर", "अक्तूबर"], ["नवंबर", "नवम्बर"],
                             ["दिसंबर", "दिसम्बर"]]):
    MONTHS.update({n: _i + 1 for n in _names})
_W = r"[A-Za-zऀ-ॿ]+"


def find_dates(text, year=2026):
    t = str(text).translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    found = set()

    def add(y, m, d):
        try:
            found.add(date(int(y or year), int(m), int(d)))
        except (ValueError, TypeError):
            pass
    for y, m, d in re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        add(y, m, d)
    for d, w, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_W})\.?,?\s*(\d{{4}})?", t):
        if w.lower() in MONTHS:
            add(y, MONTHS[w.lower()], d)
    for w, d, y in re.findall(rf"({_W})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(\d{{4}})?", t):
        if w.lower() in MONTHS:
            add(y, MONTHS[w.lower()], d)
    for d, m, y in re.findall(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b", t):
        add(y, m, d)
    return sorted(found)


def date_table(start, n_business=10, calendar=(7, 10, 15, 30)):
    bus, d = [], start
    while len(bus) < n_business:
        d += timedelta(days=1)
        if d.weekday() != 6 and d.isoformat() not in HOLIDAYS:
            bus.append(f"+{len(bus) + 1}={d.day} {d:%b}")
    cal = ", ".join(f"+{c}={(start + timedelta(days=c)).day} {(start + timedelta(days=c)):%b}" for c in calendar)
    return f"From {start.day} {start:%B %Y} ({start:%a}): business days {', '.join(bus)} | calendar days {cal}"


RULES = """- Use the date table; never count days yourself. Business days already skip Sundays and holidays.
- For "by when / kab tak / कब तक" questions take the LAST number of the range in the section (e.g. 5-7 -> +7)
  and give that ONE date with the year: "by 27 October 2026". Do not give a range of dates.
- Before "reply", add a key "calc" to the JSON showing the lookup, e.g.
  "calc": "net banking 3-5 business days -> +5 from 17 Oct -> 24 Oct". The customer never sees "calc".
- Always state the exact figure from the section that answers the question (days, months, rupee amount, date).
- Product rows give an item's return category and brand warranty; the policy section gives the rule. For
  warranty or Assured questions, give the warranty period in months from the warranty policy section; the
  policy section wins when it differs from the product row.
- If the sections do not explicitly cover the question, say you don't have that information and offer a human
  colleague. Never infer a yes or no from what the documents leave out.
- If a rule depends on a date the customer did not give, assume a recent order (last 30 days): state that rule
  with its number and mention the older rule in one short clause. Do not only ask for the date.
- decision="answer" when the customer asks about a policy or what to do (how long, can I, what happens if),
  even if you also invite them to share an order ID. Telling them an item is not returnable or a request is
  not possible is also decision="answer".
- decision="clarify" when they ask about the status of their own refund or order ("refund status", "where is
  my refund") without an order ID, payment method and start date: ask ONE question and do not quote general
  timelines.
- decision="refuse" only when they ask you to do something against policy (an exception, compensation) or for
  another person's order, address or contact details.
- decision="escalate" when the customer reports an order or payment they did not make, or the documents say a
  human must decide.
- When handing over, name a human colleague and say you will connect them
  (Hinglish: "main aapko human colleague se connect karta hoon").
- In English replies write without contractions ("will not", "do not")."""


def make_guide(question):
    today = date.fromisoformat(TODAY)
    starts = [d for d in find_dates(question) if d != today] + [today]
    tables = "\n".join("  " + date_table(d) for d in starts)
    return ("NOTES FOR THIS ANSWER (not a document; never cite this block):\n"
            f"- Today is {today.day} {today:%B %Y}. Date table:\n{tables}\n{RULES}\n")

# ------------------------------------------------------------------ answer


def rag_answer(question, history=None, k=TOP_K):
    """Step 2 answer: retrieve -> date table + rules + context -> LLM. Citations outside the context are dropped."""
    hits = retrieve(question, k)
    context = make_guide(question) + "\n" + format_context(hits)
    out = ask(question, context=context, history=history)
    ids = list(dict.fromkeys(h["id"] for h in hits))
    out["retrieved"] = ids
    out["cited_raw"] = list(out.get("used_sections") or [])
    out["used_sections"] = [s for s in out["cited_raw"] if s in ids]
    out["context_chars"] = len(context)
    return out
