import os, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/T-pro-it-2.0"))
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.lora)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.lora).save_pretrained(args.out)
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()
