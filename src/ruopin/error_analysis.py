import sys, json, argparse
from collections import Counter
from .score import load_jsonl
from .official_eval import convert_opinion_to_tuple, sent_tuples_in_list
from .stats import src_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    gold = {s["sent_id"]: s for s in load_jsonl(args.gold)}
    pred = {s["sent_id"]: s for s in load_jsonl(args.pred)}

    fn_by_src, fn_pol, fp = Counter(), 0, 0
    fn, tp = 0, 0
    fn_discont = 0
    for sid, g in gold.items():
        p = pred.get(sid, {"text": g["text"], "opinions": []})
        gt = convert_opinion_to_tuple(g)
        pt = convert_opinion_to_tuple(p)
        for o, t in zip(g["opinions"], gt):
            if sent_tuples_in_list(t, pt, keep_polarity=True):
                tp += 1
            else:
                fn += 1
                fn_by_src[src_type(o)] += 1
                if len(o["Polar_expression"][0]) > 1:
                    fn_discont += 1
                if sent_tuples_in_list(t, pt, keep_polarity=False):
                    fn_pol += 1  # span matched but polarity wrong
        for t in pt:
            if not sent_tuples_in_list(t, gt, keep_polarity=True):
                fp += 1
    print(json.dumps({
        "true_positive": tp, "false_negative": fn, "false_positive": fp,
        "fn_polarity_flip": fn_pol, "fn_by_source_type": dict(fn_by_src),
        "fn_discontinuous_expr": fn_discont,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
