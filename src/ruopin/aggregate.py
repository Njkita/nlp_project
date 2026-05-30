import os, json, argparse
from collections import defaultdict
from .serialize import _extract_json_array, _norm_polarity, parse_output
from .score import load_jsonl, score


def _norm(s):
    return " ".join(str(s).lower().split())


def _item_key(it):
    pol = _norm_polarity(it.get("polarity"))
    expr = it.get("expression", "")
    if isinstance(expr, list):
        expr = " ".join(expr)
    return (_norm(it.get("holder", "NULL")), _norm(it.get("target", "")), _norm(expr), pol)


def aggregate_sentence(text, outputs, min_votes):
    counts = defaultdict(int)
    rep = {}
    for out in outputs:
        items = _extract_json_array(out) or []
        seen = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            k = _item_key(it)
            if k[3] is None or not k[1] or not k[2]:
                continue
            if k in seen:
                continue
            seen.add(k)
            counts[k] += 1
            rep.setdefault(k, it)
    chosen = [rep[k] for k, c in counts.items() if c >= min_votes]
    return parse_output(text, json.dumps(chosen, ensure_ascii=False))


def build_preds(gen_rows, min_votes):
    preds = []
    for r in gen_rows:
        ops = aggregate_sentence(r["text"], r["outputs"], min_votes)
        preds.append({"sent_id": r["sent_id"], "text": r["text"], "opinions": ops})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generations jsonl from infer")
    ap.add_argument("--out", required=True, help="predictions jsonl (official format)")
    ap.add_argument("--gold", default=None, help="if given, also print F1")
    ap.add_argument("--min_votes", type=int, default=1)
    ap.add_argument("--sweep", action="store_true", help="if gold given, sweep min_votes")
    args = ap.parse_args()

    gen = load_jsonl(args.gen)
    n_samples = max(len(r["outputs"]) for r in gen)

    if args.sweep and args.gold:
        gold = load_jsonl(args.gold)
        best = (-1, None)
        for v in range(1, n_samples + 1):
            f1 = score(gold, build_preds(gen, v))
            print(f"min_votes={v}: f1={f1:.4f}")
            if f1 > best[0]:
                best = (f1, v)
        print(f"best min_votes={best[1]} f1={best[0]:.4f}")
        args.min_votes = best[1]

    preds = build_preds(gen, args.min_votes)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.gold:
        print(f"final f1 (min_votes={args.min_votes}): {score(load_jsonl(args.gold), preds):.4f}")
    print(f"wrote preds -> {args.out}")


if __name__ == "__main__":
    main()
