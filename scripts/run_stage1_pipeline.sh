#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/qwen-med-align-auto}"
PY_BIN="${PYTHON:-/root/miniconda3/bin/python}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/qwen-med-runs}"

export PATH="/root/miniconda3/bin:${PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pipeline" runs/predictions runs/metrics reports

PIPELINE_LOG="$RUN_ROOT/logs/stage1_pipeline_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$PIPELINE_LOG" "$RUN_ROOT/logs/stage1_pipeline_latest.log"

run_step() {
  local name="$1"
  shift
  echo
  echo "========== [$name] $(date '+%F %T') =========="
  "$@"
  echo "========== [$name done] $(date '+%F %T') =========="
}

{
  echo "[pipeline] project=$PROJECT_DIR"
  echo "[pipeline] run_root=$RUN_ROOT"
  echo "[pipeline] log=$PIPELINE_LOG"
  nvidia-smi

  run_step "preflight" bash scripts/preflight_gpu.sh
  run_step "sft-stage1-lora" bash scripts/run_train.sh sft-stage1-lora
  run_step "dpo-stage1-lora" bash scripts/run_train.sh dpo-stage1-lora

  run_step "eval-generate-cmb-val" "$PY_BIN" scripts/generate_eval_predictions.py \
    --eval data/eval/cmb_val_choice_eval.jsonl \
    --task choice \
    --adapter /root/autodl-tmp/qwen-med-runs/general-med-dpo-lora \
    --out runs/predictions/cmb_val_choice_stage1_dpo.jsonl \
    --limit 280 \
    --max-new-tokens 16

  run_step "eval-score-cmb-val" "$PY_BIN" scripts/eval_predictions.py \
    --predictions runs/predictions/cmb_val_choice_stage1_dpo.jsonl \
    --task choice \
    --out runs/metrics/cmb_val_choice_stage1_dpo.json

  run_step "report" "$PY_BIN" scripts/make_project_report.py
  run_step "storage" bash scripts/storage_guard.sh
  echo "[pipeline] all done $(date '+%F %T')"
} 2>&1 | tee "$PIPELINE_LOG"
