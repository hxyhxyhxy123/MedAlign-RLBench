#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/qwen-med-align-auto}"
PY_BIN="${PYTHON:-/root/miniconda3/bin/python}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/qwen-med-runs}"
MODE="${1:-}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PATH="/root/miniconda3/bin:${PATH:-}"
export PYTHONPATH="$PROJECT_DIR/baselines/LLaMA-Factory/src:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/logs"

case "$MODE" in
  sft-stage1-lora)
    CONFIG="configs/llamafactory/qwen25_3b_sft_lora.yaml"
    ;;
  sft-stage1-lora-zero2)
    CONFIG="configs/llamafactory/qwen25_3b_sft_lora_zero2.yaml"
    ;;
  sft-stage1-qlora)
    CONFIG="configs/llamafactory/qwen25_3b_sft_qlora.yaml"
    ;;
  dpo-stage1-lora)
    CONFIG="configs/llamafactory/qwen25_3b_dpo_lora.yaml"
    ;;
  dpo-stage1-lora-zero2)
    CONFIG="configs/llamafactory/qwen25_3b_dpo_lora_zero2.yaml"
    ;;
  redflag-sft)
    CONFIG="configs/llamafactory/qwen25_3b_redflag_sft_lora.yaml"
    ;;
  redflag-dpo)
    CONFIG="configs/llamafactory/qwen25_3b_redflag_dpo_lora.yaml"
    ;;
  redflag-sft-aug)
    CONFIG="configs/llamafactory/qwen25_3b_redflag_sft_lora_aug.yaml"
    ;;
  redflag-dpo-aug)
    CONFIG="configs/llamafactory/qwen25_3b_redflag_dpo_lora_aug.yaml"
    ;;
  sft-complete-lora)
    CONFIG="configs/llamafactory/qwen25_3b_sft_lora_complete.yaml"
    ;;
  sft-complete-qlora)
    CONFIG="configs/llamafactory/qwen25_3b_sft_qlora_complete.yaml"
    ;;
  dpo-complete-lora)
    CONFIG="configs/llamafactory/qwen25_3b_dpo_lora_complete.yaml"
    ;;
  *)
    echo "Usage: bash scripts/run_train.sh {sft-stage1-lora|sft-stage1-lora-zero2|sft-stage1-qlora|dpo-stage1-lora|dpo-stage1-lora-zero2|redflag-sft|redflag-dpo|redflag-sft-aug|redflag-dpo-aug|sft-complete-lora|sft-complete-qlora|dpo-complete-lora}"
    exit 2
    ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[run] usable nvidia-smi/GPU not found. Please switch to GPU mode before training."
  exit 3
fi

echo "[run] mode=$MODE"
echo "[run] config=$CONFIG"
nvidia-smi

if command -v llamafactory-cli >/dev/null 2>&1; then
  TRAIN_CMD=(llamafactory-cli train "$CONFIG")
else
  TRAIN_CMD=("$PY_BIN" baselines/LLaMA-Factory/src/train.py "$CONFIG")
fi

LOG="$RUN_ROOT/logs/${MODE}_$(date +%Y%m%d_%H%M%S).log"
echo "[run] log=$LOG"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG"
