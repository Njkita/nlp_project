#!/bin/bash
# Full pipeline for RuOpinionNE-2024. Run from repo root with the venv active.
set -e
export PYTHONPATH=src
MODEL=~/models/T-pro-it-2.0
VAL=RuOpinionNE-2024/validation_labeled.jsonl

# 1. training data
python3 -m ruopin.build_sft --train RuOpinionNE-2024/train.jsonl --out_train data/sft_train.jsonl

# 2. baselines (no training): zero-shot and few-shot of the base model
python3 -m ruopin.infer_vllm --model $MODEL --data $VAL --out outputs/zs.jsonl --n_shots 0 --temperature 0
python3 -m ruopin.aggregate --gen outputs/zs.jsonl --out outputs/zs_pred.jsonl --gold $VAL --min_votes 1
python3 -m ruopin.infer_vllm --model $MODEL --data $VAL --out outputs/fs.jsonl --n_shots 3 --temperature 0
python3 -m ruopin.aggregate --gen outputs/fs.jsonl --out outputs/fs_pred.jsonl --gold $VAL --min_votes 1

# 3. QLoRA fine-tune
python3 -m ruopin.train_qlora --model $MODEL --data data/sft_train.jsonl --out out/tpro-lora \
  --epochs 3 --lr 1e-4 --bs 4 --accum 4 --r 32 --alpha 64

# 4. merge adapter into bf16 for fast serving
python3 -m ruopin.merge_lora --base $MODEL --lora out/tpro-lora --out out/tpro-merged

# 5. tuned inference: greedy and self-consistency (5 samples), then aggregate sweep
python3 -m ruopin.infer_vllm --model out/tpro-merged --data $VAL --out outputs/ft_greedy.jsonl --temperature 0
python3 -m ruopin.aggregate --gen outputs/ft_greedy.jsonl --out outputs/ft_greedy_pred.jsonl --gold $VAL --min_votes 1
python3 -m ruopin.infer_vllm --model out/tpro-merged --data $VAL --out outputs/ft_sc.jsonl --samples 5 --temperature 0.6
python3 -m ruopin.aggregate --gen outputs/ft_sc.jsonl --out outputs/ft_sc_pred.jsonl --gold $VAL --sweep
