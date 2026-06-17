#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-med-align-auto

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
RUN_DIR=/root/autodl-tmp/qwen-med-runs/general-med-choice-answer-dpo-lora
EVAL=data/eval/cmb_test_choice_3000_random_noleak.jsonl
PRED_DIR=runs/predictions/cmb_test_3000_random_noleak_v6_checkpoints
METRIC_DIR=runs/metrics/cmb_test_3000_random_noleak_v6_checkpoints
mkdir -p runs/logs "$PRED_DIR" "$METRIC_DIR"

LOCK=runs/logs/choice_answer_checkpoint_sweep_v6.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[ckpt-sweep-v6] another runner is already active"
  exit 0
fi

wait_for_gpu() {
  local max_wait_seconds=${1:-21600}
  local waited=0
  while true; do
    if /root/miniconda3/bin/python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
    then
      echo "[ckpt-sweep-v6] CUDA is available $(date '+%F %T')"
      return 0
    fi
    if [ "$waited" -ge "$max_wait_seconds" ]; then
      echo "[ckpt-sweep-v6] CUDA still unavailable after ${max_wait_seconds}s; exiting $(date '+%F %T')"
      return 1
    fi
    echo "[ckpt-sweep-v6] waiting for CUDA... waited=${waited}s $(date '+%F %T')"
    sleep 60
    waited=$((waited + 60))
  done
}

eval_one() {
  local step=$1
  local adapter="$RUN_DIR/checkpoint-$step"
  local name="checkpoint_$step"
  echo "[ckpt-sweep-v6] evaluating $name $(date '+%F %T')"
  /root/miniconda3/bin/python -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$adapter" \
    --eval "$EVAL" \
    --task choice \
    --out "$PRED_DIR/$name.jsonl" \
    --start 0 \
    --limit 3000 \
    --max-new-tokens 16 \
    --progress-every 100 \
    --flush-every 1 \
    --resume
  /root/miniconda3/bin/python scripts/eval_choice_predictions_robust.py \
    --predictions "$PRED_DIR/$name.jsonl" \
    --out "$METRIC_DIR/$name.json"
}

wait_for_gpu 21600

for step in 3000 4000 5000 6000 7000 7500; do
  eval_one "$step"
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
metric_dir = Path("runs/metrics/cmb_test_3000_random_noleak_v6_checkpoints")
summary = {}
for path in sorted(metric_dir.glob("checkpoint_*.json")):
    m = json.loads(path.read_text(encoding="utf-8"))
    summary[path.stem] = {
        "accuracy": m.get("accuracy"),
        "correct": m.get("correct"),
        "total": m.get("total"),
        "single_accuracy": (m.get("single_choice") or {}).get("accuracy"),
        "multi_accuracy": (m.get("multi_choice") or {}).get("accuracy"),
        "invalid": m.get("invalid_prediction_count"),
    }
sft = json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v4/sft.json").read_text(encoding="utf-8"))
best_name, best = max(summary.items(), key=lambda item: item[1]["accuracy"])
out = {
    "baseline_sft_accuracy": sft["accuracy"],
    "best_checkpoint": best_name,
    "best_accuracy": best["accuracy"],
    "delta_vs_sft_pp": round((best["accuracy"] - sft["accuracy"]) * 100, 4),
    "summary": summary,
}
(metric_dir / "checkpoint_sweep_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

echo "[ckpt-sweep-v6] complete $(date '+%F %T')"
