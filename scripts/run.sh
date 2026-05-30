#!/bin/bash
# RuOpinionNE-2024 pipeline. Run from repo root with the venv active.
# Data must be cloned next to the repo:
#   git clone https://github.com/dialogue-evaluation/RuOpinionNE-2024
# MODEL points at a local copy of t-tech/T-pro-it-2.0.
set -e
export PYTHONPATH=src
MODEL=${MODEL:-$HOME/models/T-pro-it-2.0}
VAL=RuOpinionNE-2024/validation_labeled.jsonl
mkdir -p data outputs out

# Table 1: dataset statistics
python3 -m ruopin.stats RuOpinionNE-2024/train.jsonl "$VAL"

# training data
python3 -m ruopin.build_sft --train RuOpinionNE-2024/train.jsonl --out_train data/sft_train.jsonl

# Table 2, baselines: same base model without fine-tuning
python3 -m ruopin.infer_vllm --model "$MODEL" --data "$VAL" --out outputs/zs.jsonl --n_shots 0 --temperature 0 --quant bitsandbytes
python3 -m ruopin.aggregate --gen outputs/zs.jsonl --out outputs/zs_pred.jsonl --gold "$VAL" --min_votes 1
python3 -m ruopin.infer_vllm --model "$MODEL" --data "$VAL" --out outputs/fs.jsonl --n_shots 3 --temperature 0 --quant bitsandbytes
python3 -m ruopin.aggregate --gen outputs/fs.jsonl --out outputs/fs_pred.jsonl --gold "$VAL" --min_votes 1

# QLoRA fine-tune; an adapter is saved after every epoch under out/tpro-lora/checkpoint-*
python3 -m ruopin.train_qlora --model "$MODEL" --data data/sft_train.jsonl --out out/tpro-lora \
  --epochs 3 --lr 1e-4 --bs 2 --accum 8 --r 32 --alpha 64

# Table 3: greedy eval of every epoch (base model served with each epoch's adapter)
for ck in out/tpro-lora/checkpoint-*; do
  name=$(basename "$ck")
  python3 -m ruopin.infer_vllm --model "$MODEL" --lora "$ck" --data "$VAL" --out "outputs/$name.jsonl" --temperature 0 --quant bitsandbytes
  python3 -m ruopin.aggregate --gen "outputs/$name.jsonl" --out "outputs/${name}_pred.jsonl" --gold "$VAL" --min_votes 1
done

# last epoch: greedy is covered above; self-consistency with a min_votes sweep
BEST=$(ls -d out/tpro-lora/checkpoint-* | sort -t- -k2 -n | tail -1)
python3 -m ruopin.infer_vllm --model "$MODEL" --lora "$BEST" --data "$VAL" --out outputs/sc.jsonl --samples 5 --temperature 0.6 --quant bitsandbytes
python3 -m ruopin.aggregate --gen outputs/sc.jsonl --out outputs/sc_pred.jsonl --gold "$VAL" --sweep

# error breakdown on the best prediction
python3 -m ruopin.error_analysis --gold "$VAL" --pred outputs/sc_pred.jsonl
