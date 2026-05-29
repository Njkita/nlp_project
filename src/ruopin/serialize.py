import json
import re
from .official_eval import tk, convert_char_offsets_to_token_idxs, UNKN_ORIGIN

AUTHOR = "AUTHOR"
NULL = "NULL"


def _holder_text(source):
    head = source[0]
    if head == [AUTHOR]:
        return AUTHOR
    if head == [NULL]:
        return NULL
    return " ".join(head)


def opinions_to_target(opinions):
    items = []
    for o in opinions:
        items.append({
            "holder": _holder_text(o["Source"]),
            "target": " ".join(o["Target"][0]),
            "expression": o["Polar_expression"][0],
            "polarity": o["Polarity"],
        })
    return json.dumps(items, ensure_ascii=False)


def _find_offset(text, span, cursor):
    if not span:
        return None
    pos = text.find(span, cursor)
    if pos < 0:
        pos = text.find(span)
    if pos < 0:
        low = text.lower().find(span.lower())
        if low < 0:
            return None
        pos = low
    return pos, pos + len(span)


def _make_source(text, holder):
    if holder == AUTHOR:
        return [[AUTHOR], [NULL]]
    if holder in (NULL, "", None):
        return [[NULL], ["0:0"]]
    off = _find_offset(text, holder, 0)
    if off is None:
        return [[NULL], ["0:0"]]
    return [[holder], [f"{off[0]}:{off[1]}"]]


def _make_spanfield(text, value):
    if isinstance(value, str):
        value = [value]
    texts, offs = [], []
    cursor = 0
    for frag in value:
        frag = frag.strip()
        if not frag:
            continue
        off = _find_offset(text, frag, cursor)
        if off is None:
            continue
        texts.append(frag)
        offs.append(f"{off[0]}:{off[1]}")
        cursor = off[1]
    if not texts:
        return None
    return [texts, offs]


_POLARITY_MAP = {
    "POS": "POS", "POSITIVE": "POS", "ПОЗ": "POS", "ПОЛОЖИТЕЛЬНАЯ": "POS", "POZ": "POS",
    "NEG": "NEG", "NEGATIVE": "NEG", "НЕГ": "NEG", "ОТРИЦАТЕЛЬНАЯ": "NEG",
}


def _norm_polarity(p):
    if not isinstance(p, str):
        return None
    return _POLARITY_MAP.get(p.strip().upper())


def _extract_json_array(s):
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                chunk = s[start:i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    chunk2 = re.sub(r",\s*([\]}])", r"\1", chunk)
                    try:
                        return json.loads(chunk2)
                    except json.JSONDecodeError:
                        return None
    return None


def _token_sets(text, opinion):
    token_offsets = tk(text)
    src_idxs = opinion["Source"][1]
    holder = frozenset([AUTHOR]) if src_idxs[0] == UNKN_ORIGIN \
        else convert_char_offsets_to_token_idxs(src_idxs, token_offsets)
    target = convert_char_offsets_to_token_idxs(opinion["Target"][1], token_offsets)
    exp = convert_char_offsets_to_token_idxs(opinion["Polar_expression"][1], token_offsets)
    return holder, target, exp, opinion["Polarity"]


def sanitize(text, opinions):
    """Drop duplicates and conflicting tuples that the official scorer rejects."""
    kept, kept_sets = [], []
    for op in opinions:
        h, t, e, p = _token_sets(text, op)
        bad = False
        for h2, t2, e2, p2 in kept_sets:
            if h == h2 and t == t2 and p == p2:
                if e == e2 or len(e.intersection(e2)) > 0:
                    bad = True
                    break
        if bad:
            continue
        kept.append(op)
        kept_sets.append((h, t, e, p))
    return kept


def parse_output(text, output_str):
    items = _extract_json_array(output_str)
    if not items:
        return []
    opinions = []
    for it in items:
        if not isinstance(it, dict):
            continue
        pol = _norm_polarity(it.get("polarity"))
        if pol is None:
            continue
        target = _make_spanfield(text, it.get("target", ""))
        expr = _make_spanfield(text, it.get("expression", ""))
        if target is None or expr is None:
            continue
        source = _make_source(text, it.get("holder", NULL))
        opinions.append({
            "Source": source,
            "Target": target,
            "Polar_expression": expr,
            "Polarity": pol,
        })
    return sanitize(text, opinions)
