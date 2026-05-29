import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ruopin.serialize import opinions_to_target, parse_output
from ruopin.score import load_jsonl, score

# lossless check: gold -> serialize -> parse -> score should give F1 ~1.0

def main(path):
    gold = load_jsonl(path)
    preds = []
    drops = 0
    for s in gold:
        tgt = opinions_to_target(s["opinions"])
        rebuilt = parse_output(s["text"], tgt)
        drops += max(0, len(s["opinions"]) - len(rebuilt))
        preds.append({"sent_id": s["sent_id"], "text": s["text"], "opinions": rebuilt})
    f1 = score(gold, preds)
    n_op = sum(len(s["opinions"]) for s in gold)
    print(f"sentences={len(gold)} gold_tuples={n_op} dropped_in_roundtrip={drops}")
    print(f"roundtrip f1: {f1:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "RuOpinionNE-2024/validation_labeled.jsonl")
