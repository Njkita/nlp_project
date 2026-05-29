import os, json, argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from .prompts import build_messages
from .score import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_shots", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_mem_util", type=float, default=0.5)
    ap.add_argument("--quant", default="bitsandbytes")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load_jsonl(args.data)
    prompts = []
    for s in rows:
        msgs = build_messages(s["text"], n_shots=args.n_shots)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False))

    kw = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True)
    if args.quant:
        kw["quantization"] = args.quant
        if args.quant == "bitsandbytes":
            kw["load_format"] = "bitsandbytes"
    if args.lora:
        kw["enable_lora"] = True
        kw["max_lora_rank"] = 64
    llm = LLM(**kw)

    sp = SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop=["<|im_end|>"],
                        seed=None if args.temperature > 0 else 0)
    lora_req = LoRARequest("ad", 1, args.lora) if args.lora else None
    res = llm.generate(prompts, sp, lora_request=lora_req)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s, r in zip(rows, res):
            outs = [o.text for o in r.outputs]
            f.write(json.dumps({"sent_id": s["sent_id"], "text": s["text"], "outputs": outs},
                               ensure_ascii=False) + "\n")
    print(f"wrote generations -> {args.out}")


if __name__ == "__main__":
    main()
