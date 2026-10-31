
import re
import sys
import pandas as pd


from config import KB_DIR, DATA_DIR, PRACTICE_DIR

EXPECTED_ROWS = {"customers": 300, "orders": 1000, "order_items": 1783, "products": 70,
                 "pincodes": 36, "holidays": 4, "support_logs": 244}
PKS = {"customers": ["customer_id"], "orders": ["order_id"], "order_items": ["order_id", "line_no"],
       "products": ["sku"], "pincodes": ["pincode"], "holidays": ["date"], "support_logs": ["log_id"]}
FKS = [("orders", "customer_id", "customers", "customer_id"),
       ("order_items", "order_id", "orders", "order_id"),
       ("order_items", "sku", "products", "sku"),
       ("orders", "delivery_pincode", "pincodes", "pincode"),
       ("customers", "pincode", "pincodes", "pincode")]
SECTION_RE = re.compile(r"^#{1,6}\s*\[([A-Za-z0-9_.-]+)\]\s*(.*)$", re.M)

errors, warnings = [], []


def load(name, folder=DATA_DIR):
    return pd.read_csv(folder / f"{name}.csv", dtype=str, keep_default_na=False)


def check_tables():
    t = {n: load(n) for n in EXPECTED_ROWS}
    for n, df in t.items():
        if len(df) != EXPECTED_ROWS[n]:
            errors.append(f"{n}: {len(df)} rows, expected {EXPECTED_ROWS[n]}")
        dup = df.duplicated(PKS[n]).sum()
        if dup:
            errors.append(f"{n}: {dup} duplicate primary keys on {PKS[n]}")
    for child, ccol, parent, pcol in FKS:
        vals = t[child][ccol]
        missing = set(vals[vals != ""]) - set(t[parent][pcol])
        if missing:
            errors.append(f"FK {child}.{ccol} -> {parent}.{pcol}: {len(missing)} orphans, e.g. {sorted(missing)[:3]}")

    o, it = t["orders"], t["order_items"]
    # format checks
    bad_ids = o[~o.order_id.str.fullmatch(r"\d{8}")]
    if len(bad_ids):
        errors.append(f"orders: {len(bad_ids)} order_ids are not 8 digits")
    bad_pin = o[~o.delivery_pincode.str.fullmatch(r"\d{6}")]
    if len(bad_pin):
        errors.append(f"orders: {len(bad_pin)} delivery pincodes are not 6 digits")
    # return line must match return_sku
    r = o[o.return_sku != ""].merge(it, left_on=["order_id", "return_line_no"],
                                    right_on=["order_id", "line_no"], how="left")
    mism = r[r.return_sku != r.sku]
    if len(mism):
        errors.append(f"orders: {len(mism)} returns where return_line_no SKU != return_sku")
    # subtotal sanity (warning only — coupons/fees may explain gaps)
    it2 = it.assign(v=it.qty.astype(int) * it.unit_price_inr.astype(float)).groupby("order_id").v.sum()
    sub = o.set_index("order_id").subtotal_inr.replace("", "0").astype(float)
    off = (sub - it2.reindex(sub.index).fillna(0)).abs() > 1
    if off.sum():
        warnings.append(f"orders: {off.sum()} subtotals differ from sum(order_items) by > ₹1")
    print("Order status distribution:\n", o.status.value_counts().to_string(), "\n")
    return t


def parse_kb_header(text):
    """Header is a key: value block (optionally fenced by ---) at the top of each file."""
    head = text.split("\n## ", 1)[0]
    meta = {}
    for line in head.splitlines():
        m = re.match(r"^\s*[-*]?\s*\**([a-z_]+)\**\s*:\s*(.+?)\s*$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip("`\"' ")
    return meta


def check_kb():
    files = sorted(KB_DIR.glob("*.md"))
    if len(files) != 21:
        errors.append(f"kb: {len(files)} markdown files, expected 21")
    all_ids = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        meta = parse_kb_header(text)
        for key in ("doc_id", "status", "effective_from", "doc_type"):
            if key not in meta:
                warnings.append(f"kb/{f.name}: header missing '{key}'")
        ids = SECTION_RE.findall(text)
        if not ids:
            errors.append(f"kb/{f.name}: no [SECTION-ID] headings found")
        for sid, _ in ids:
            if sid in all_ids:
                errors.append(f"kb: section {sid} in both {all_ids[sid]} and {f.name}")
            all_ids[sid] = f.name
        print(f"  {f.name:45s} {meta.get('status', '?'):11s} {meta.get('effective_from', '?'):11s} {len(ids)} sections")
    return all_ids


def check_practice(section_ids, products):
    skus = set(products.sku)
    q = load("questions", PRACTICE_DIR)
    c = load("conversations", PRACTICE_DIR)
    if len(q) != 52: errors.append(f"questions: {len(q)} rows, expected 52")
    if len(c) != 96: errors.append(f"conversations: {len(c)} rows, expected 96")
    if c.conversation_id.nunique() != 23: errors.append("conversations: expected 23 conversation_ids")
    for name, df in (("questions", q), ("conversations", c)):
        for _, row in df.iterrows():
            for ev in filter(None, (x.strip() for x in row.gold_evidence.split(";"))):
                if ev.startswith("products.csv:"):
                    if ev.split(":", 1)[1] not in skus:
                        errors.append(f"{name}: unknown SKU evidence {ev}")
                elif ev not in section_ids:
                    errors.append(f"{name}: gold_evidence {ev} not found in kb/")
    print("\nQuestion decisions:\n", q.expected_decision.value_counts().to_string())
    print("Question languages:\n", q.language.value_counts().to_string())
    # turns consecutive
    for cid, g in c.groupby("conversation_id"):
        turns = sorted(g.turn.astype(int))
        if turns != list(range(1, len(turns) + 1)):
            errors.append(f"conversation {cid}: turns not consecutive {turns}")


if __name__ == "__main__":
    tables = check_tables()
    print("KB files:")
    ids = check_kb()
    check_practice(ids, tables["products"])
    print(f"\n{len(warnings)} warnings"); [print("  WARN", w) for w in warnings]
    print(f"{len(errors)} errors"); [print("  ERR ", e) for e in errors]
    sys.exit(1 if errors else 0)
