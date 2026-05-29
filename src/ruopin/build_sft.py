import json
import argparse
from .prompts import build_train_messages, build_messages
from .score import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="RuOpinionNE-2024/train.jsonl")
    ap.add_argument("--out_train", default="data/sft_train.jsonl")
    ap.add_argument("--extra", default=None, help="optional extra labeled jsonl to fold into training")
    args = ap.parse_args()

    rows = load_jsonl(args.train)
    if args.extra:
        rows += load_jsonl(args.extra)

    with open(args.out_train, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps({"messages": build_train_messages(s)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} sft rows -> {args.out_train}")


if __name__ == "__main__":
    main()
