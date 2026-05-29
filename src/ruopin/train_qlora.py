import os, sys, json, argparse
import torch
from dataclasses import dataclass
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE = -100


def render(tokenizer, messages):
    full = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False, enable_thinking=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    return full, prompt


def build_dataset(path, tokenizer, max_len):
    feats = []
    n_trunc = 0
    for line in open(path, encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        full, prompt = render(tokenizer, msgs)
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("prompt is not a prefix of full render; check chat template")
        labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(full_ids) > max_len:
            n_trunc += 1
            full_ids = full_ids[:max_len]
            labels = labels[:max_len]
        feats.append({"input_ids": full_ids, "labels": labels})
    print(f"dataset: {len(feats)} examples, truncated={n_trunc}")
    return feats


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * n)
            lab.append(b["labels"] + [IGNORE] * n)
            att.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/T-pro-it-2.0"))
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="out/tpro-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation=args.attn)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data = build_dataset(args.data, tok, args.max_len)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.0, bf16=True, logging_steps=10, save_strategy="epoch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], optim="paged_adamw_8bit", max_grad_norm=0.3, seed=args.seed)

    trainer = Trainer(model=model, args=targs, train_dataset=data,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
