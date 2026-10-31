"""Scoring for the practice questions. Same code scores Step 1 and Step 2, so results are comparable."""
import re
import unicodedata
from datetime import datetime

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MONTHS_EN = ["january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december"]
MONTHS_HI = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलई", "अगस्त",
             "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14,
             "fifteen": 15, "thirty": 30, "forty-eight": 48, "seventy-two": 72}



    
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).translate(DEV_DIGITS).lower()
    s = s.replace("फरवरी", "फ़रवरी").replace("सितम्बर", "सितंबर")
    s = s.replace("₹", " ").replace("rs.", " ").replace("inr", " ")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)             # 1,299 -> 1299
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)    # 27th -> 27
    for w, n in NUM_WORDS.items():
        s = re.sub(rf"\b{w}\b", str(n), s)
    return re.sub(r"\s+", " ", s).strip()


def date_variants(text):
    """ISO date or '27 October 2026' -> many equivalent spellings (English + Hindi months)."""
    t = norm(text)
    d = None
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d %Y", "%d %b %Y", "%d/%m/%Y"):
        try:
            d = datetime.strptime(t.replace(",", "").title() if "%B" in fmt or "%b" in fmt else t, fmt)
            break
        except ValueError:
            pass
    if d is None:
        return [t]
    mi = d.month - 1
    return list({norm(v) for v in [
        d.strftime("%Y-%m-%d"), f"{d.day} {MONTHS_EN[mi]} {d.year}", f"{MONTHS_EN[mi]} {d.day} {d.year}",
        f"{MONTHS_EN[mi]} {d.day}, {d.year}", f"{d.day} {MONTHS_EN[mi][:3]} {d.year}",
        f"{d.day} {MONTHS_HI[mi]} {d.year}", f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.day} {MONTHS_EN[mi]}", f"{d.day} {MONTHS_HI[mi]}"]})


def contains(reply_n, phrase):
    for v in date_variants(phrase):
        if not v:
            continue
        if re.fullmatch(r"\d+", v):                   # bare number: '5' must not match '15'
            if re.search(rf"(?<!\d){v}(?!\d)", reply_n):
                return True
        elif v in reply_n:
            return True
    return False


def split(cell):
    return [x.strip() for x in str(cell or "").split(" ; ") if x.strip()]


def score(row, out):
    """row: a questions.csv row. out: dict from ask()/rag_answer()."""
    r = norm(out.get("reply", ""))
    facts = split(row.must_include)
    hits = [any(contains(r, alt.strip()) for alt in f.split(" / ")) for f in facts]
    bad = [p for p in split(row.must_not_include) if contains(r, p)]
    acceptable = {a.strip().lower() for a in re.split(r"[;,/]", row.acceptable_decisions or row.expected_decision)
                  if a.strip()}
    dec_ok = str(out.get("decision") or "").lower() in acceptable
    return {"decision_ok": dec_ok,
            "fact_recall": sum(hits) / len(hits) if facts else 1.0,
            "all_facts": all(hits),
            "missing": [f for f, h in zip(facts, hits) if not h],
            "violation": bool(bad), "violations": bad,
            "correct": dec_ok and all(hits) and not bad}


def retrieval_metrics(retrieved, gold, cited=None):
    g = set(gold)
    if not g:
        return {}
    found = [i for i in retrieved if i in g]
    out = {"recall@k": len(set(found)) / len(g), "hit@k": float(bool(found)),
           "mrr": next((1 / (r + 1) for r, i in enumerate(retrieved) if i in g), 0.0)}
    if cited is not None:
        c = set(cited)
        out["cite_precision"] = len(c & g) / len(c) if c else 0.0
    return out
