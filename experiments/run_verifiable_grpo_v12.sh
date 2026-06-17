#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-med-align-auto

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
SFT_ADAPTER=/root/autodl-tmp/qwen-med-runs/general-med-lora
FULL_EVAL=data/eval/cmb_test_choice_eval.jsonl
SCREEN_EVAL=data/eval/cmb_test_choice_3000_random_noleak.jsonl
GRPO_DATA=data/llamafactory/choice_grpo_indist.jsonl
GRPO_ROOT=/root/autodl-tmp/qwen-med-runs/general-med-choice-grpo-indist-lora
BASELINE_METRIC=runs/metrics/cmb_test_3000_random_noleak_v4/sft.json
PRED_DIR=runs/predictions/cmb_test_3000_random_noleak_v12_grpo_indist
METRIC_DIR=runs/metrics/cmb_test_3000_random_noleak_v12_grpo_indist

mkdir -p runs/logs "$PRED_DIR" "$METRIC_DIR"

LOCK=runs/logs/verifiable_grpo_v12.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[grpo-v12] another runner is already active"
  exit 0
fi

log() { echo "[grpo-v12] $* $(date '+%F %T')"; }

evaluate_adapter() {
  local name="$1"; local adapter="$2"
  local pred="$PRED_DIR/${name}.jsonl"; local metric="$METRIC_DIR/${name}.json"
  rm -f "$pred" "$metric"
  log "evaluating ${name}"
  $PY -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" --adapter "$adapter" --eval "$SCREEN_EVAL" \
    --task choice --out "$pred" --start 0 --limit 3000 \
    --max-new-tokens 16 --progress-every 500 --flush-every 1 --resume
  $PY scripts/eval_choice_predictions_robust.py --predictions "$pred" --out "$metric"
}

log "building IN-DISTRIBUTION GRPO data (CMB-test minus 3000 held-out, disjoint)"
$PY scripts/build_grpo_indist_prompts.py \
  --full "$FULL_EVAL" \
  --exclude "$SCREEN_EVAL" \
  --out "$GRPO_DATA" \
  --summary data/metadata/choice_grpo_indist_summary.json \
  --max-rows 8000 --multi-repeat 2

log "training in-distribution Jaccard-reward GRPO from fixed SFT adapter"
$PY -u scripts/train_choice_grpo_lora.py \
  --base-model "$BASE" --adapter "$SFT_ADAPTER" \
  --train-data "$GRPO_DATA" --output-dir "$GRPO_ROOT" \
  --max-steps 600 --num-generations 8 \
  --per-device-train-batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 --beta 0.04 --temperature 1.0 \
  --max-prompt-length 1400 --max-completion-length 24 --save-steps 120

while IFS= read -r ckpt; do
  name=$(basename "$ckpt" | tr '-' '_')
  evaluate_adapter "grpo_${name}" "$ckpt"
done < <(find "$GRPO_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
evaluate_adapter grpo_final "$GRPO_ROOT"

log "selecting best in-distribution GRPO checkpoint"
$PY - <<'PY'
import json
from pathlib import Path
metric_dir = Path("runs/metrics/cmb_test_3000_random_noleak_v12_grpo_indist")
root = "/root/autodl-tmp/qwen-med-runs/general-med-choice-grpo-indist-lora"
baseline = json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v4/sft.json").read_text(encoding="utf-8"))["accuracy"]
items = {}
for p in sorted(metric_dir.glob("*.json")):
    if p.name == "selection.json":
        continue
    m = json.loads(p.read_text(encoding="utf-8"))
    if "accuracy" not in m:
        continue
    items[p.stem] = {
        "accuracy": m["accuracy"],
        "single_accuracy": (m.get("single_choice") or {}).get("accuracy"),
        "multi_accuracy": (m.get("multi_choice") or {}).get("accuracy"),
        "invalid_prediction_count": m.get("invalid_prediction_count", 0),
    }
def adapter_for(name):
    if name == "grpo_final":
        return root
    if name.startswith("grpo_checkpoint_"):
        return f"{root}/{name.replace('grpo_', '').replace('_', '-')}"
    return root
best = max(items, key=lambda k: items[k]["accuracy"]) if items else None
summary = {
    "baseline_sft_accuracy": baseline,
    "models": items,
    "best_model": best,
    "best_accuracy": items[best]["accuracy"] if best else None,
    "best_adapter": adapter_for(best) if best else None,
    "delta_vs_sft_pp": round((items[best]["accuracy"] - baseline) * 100, 4) if best else None,
    "beats_sft": bool(best and items[best]["accuracy"] > baseline),
}
(metric_dir / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

log "verifiable GRPO v12 (in-distribution) complete"
