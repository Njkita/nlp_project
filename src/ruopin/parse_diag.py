import json
import argparse
from .serialize import parse_output
from .score import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generations jsonl (text + outputs)")
    args = ap.parse_args()

    stats = {}
    samples = 0
    for r in load_jsonl(args.gen):
        for out in r["outputs"]:
            samples += 1
            parse_output(r["text"], out, stats)
    stats["samples_parsed"] = samples
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
