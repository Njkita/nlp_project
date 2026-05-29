import sys, json, argparse
from collections import Counter
from .score import load_jsonl

AUTHOR, NULL = "AUTHOR", "NULL"


def src_type(o):
    head = o["Source"][0]
    if head == [AUTHOR]:
        return "author"
    if head == [NULL]:
        return "null"
    return "entity"


def stats(path):
    rows = load_jsonl(path)
    n_sent = len(rows)
    n_empty = sum(1 for s in rows if not s["opinions"])
    tuples = [o for s in rows for o in s["opinions"]]
    pol = Counter(o["Polarity"] for o in tuples)
    src = Counter(src_type(o) for o in tuples)
    per_sent = Counter(len(s["opinions"]) for s in rows)
    multi_expr = sum(1 for o in tuples if len(o["Polar_expression"][0]) > 1)
    return {
        "file": path, "sentences": n_sent, "empty_sentences": n_empty,
        "tuples": len(tuples), "polarity": dict(pol), "source_type": dict(src),
        "discontinuous_expressions": multi_expr,
        "tuples_per_sentence": dict(sorted(per_sent.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        print(json.dumps(stats(p), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
