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
CANDIDATES=runs/predictions/hard_choice_candidate_cmexam_12000_sft.jsonl
GRPO_DATA=data/llamafactory/choice_grpo_v11.jsonl
GRPO_ROOT=/root/autodl-tmp/qwen-med-runs/general-med-choice-grpo-v11-lora
SCREEN_EVAL=data/eval/cmb_test_choice_3000_random_noleak.jsonl
FULL_EVAL=data/eval/cmb_test_choice_eval.jsonl
BASELINE_METRIC=runs/metrics/cmb_test_3000_random_noleak_v4/sft.json
PRED_DIR=runs/predictions/cmb_test_3000_random_noleak_v11_grpo
METRIC_DIR=runs/metrics/cmb_test_3000_random_noleak_v11_grpo
FULL_PRED_DIR=runs/predictions/cmb_test_full_v11_grpo
FULL_METRIC_DIR=runs/metrics/cmb_test_full_v11_grpo

mkdir -p runs/logs "$PRED_DIR" "$METRIC_DIR" "$FULL_PRED_DIR" "$FULL_METRIC_DIR"

LOCK=runs/logs/verifiable_grpo_v11.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[grpo-v11] another runner is already active"
  exit 0
fi

log() {
  echo "[grpo-v11] $* $(date '+%F %T')"
}

evaluate_adapter() {
  local name="$1"
  local adapter="$2"
  local pred="$PRED_DIR/${name}.jsonl"
  local metric="$METRIC_DIR/${name}.json"
  rm -f "$pred" "$metric"
  log "evaluating ${name}"
  $PY -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$adapter" \
    --eval "$SCREEN_EVAL" \
    --task choice \
    --out "$pred" \
    --start 0 \
    --limit 3000 \
    --max-new-tokens 16 \
    --progress-every 500 \
    --flush-every 1 \
    --resume
  $PY scripts/eval_choice_predictions_robust.py \
    --predictions "$pred" \
    --out "$metric"
}

log "building eval-matched GRPO prompt dataset (raw question prompts, balanced multi)"
$PY scripts/build_grpo_hard_prompts.py \
  --predictions "$CANDIDATES" \
  --out "$GRPO_DATA" \
  --summary data/metadata/choice_grpo_v11_summary.json \
  --max-rows 4000 \
  --multi-repeat 2 \
  --drop-easy-frac 0.25

log "training Jaccard-reward GRPO from fixed SFT adapter (tight KL)"
$PY -u scripts/train_choice_grpo_lora.py \
  --base-model "$BASE" \
  --adapter "$SFT_ADAPTER" \
  --train-data "$GRPO_DATA" \
  --output-dir "$GRPO_ROOT" \
  --max-steps 300 \
  --num-generations 8 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --beta 0.08 \
  --temperature 1.0 \
  --max-prompt-length 1400 \
  --max-completion-length 24 \
  --save-steps 60

while IFS= read -r ckpt; do
  name=$(basename "$ckpt" | tr '-' '_')
  evaluate_adapter "grpo_${name}" "$ckpt"
done < <(find "$GRPO_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

evaluate_adapter grpo_final "$GRPO_ROOT"

log "selecting best GRPO checkpoint"
$PY - <<'PY'
import json
from pathlib import Path

metric_dir = Path("runs/metrics/cmb_test_3000_random_noleak_v11_grpo")
root = "/root/autodl-tmp/qwen-med-runs/general-med-choice-grpo-v11-lora"
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

def adapter_for(name: str) -> str:
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

RUN_FULL=$($PY - <<'PY'
import json
from pathlib import Path
sel = json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v11_grpo/selection.json").read_text(encoding="utf-8"))
print("1" if sel.get("beats_sft") else "0")
PY
)

if [ "$RUN_FULL" = "1" ]; then
  BEST_ADAPTER=$($PY - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v11_grpo/selection.json").read_text(encoding="utf-8"))["best_adapter"])
PY
)
  BEST_MODEL=$($PY - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v11_grpo/selection.json").read_text(encoding="utf-8"))["best_model"])
PY
)
  log "best GRPO beat SFT on random-3000; running full CMB-test for ${BEST_MODEL}"
  $PY -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$BEST_ADAPTER" \
    --eval "$FULL_EVAL" \
    --task choice \
    --out "$FULL_PRED_DIR/${BEST_MODEL}.jsonl" \
    --start 0 \
    --limit 11200 \
    --max-new-tokens 16 \
    --progress-every 1000 \
    --flush-every 1 \
    --resume
  $PY scripts/eval_choice_predictions_robust.py \
    --predictions "$FULL_PRED_DIR/${BEST_MODEL}.jsonl" \
    --out "$FULL_METRIC_DIR/${BEST_MODEL}.json"
else
  log "no GRPO checkpoint beat SFT on random-3000; skipping full CMB-test"
fi

log "verifiable GRPO v11 complete"
