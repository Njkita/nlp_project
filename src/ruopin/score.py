import json
import argparse
from .official_eval import do_eval_core


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score(gold, preds):
    gold = sorted(gold, key=lambda s: s["sent_id"])
    by_id = {s["sent_id"]: s for s in preds}
    aligned = []
    for s in gold:
        p = by_id.get(s["sent_id"])
        aligned.append(p if p is not None else {"sent_id": s["sent_id"], "text": s["text"], "opinions": []})
    return do_eval_core(gold=gold, preds=aligned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    f1 = score(load_jsonl(args.gold), load_jsonl(args.pred))
    print(f"f1: {f1:.4f}")


if __name__ == "__main__":
    main()
