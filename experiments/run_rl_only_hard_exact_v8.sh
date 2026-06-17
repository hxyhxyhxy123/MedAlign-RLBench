#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-med-align-auto

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

BASE=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
SFT_ADAPTER=/root/autodl-tmp/qwen-med-runs/general-med-lora
SCREEN_EVAL=data/eval/cmb_test_choice_3000_random_noleak.jsonl
FULL_EVAL=data/eval/cmb_test_choice_eval.jsonl
BASELINE_METRIC=runs/metrics/cmb_test_3000_random_noleak_v4/sft.json
SOURCE_PRED=runs/predictions/hard_choice_candidate_cmexam_12000_sft.jsonl
DATASET=data/llamafactory/choice_hard_exact_dpo.jsonl
DATA_SUMMARY=data/metadata/choice_hard_exact_dpo_summary.json
PRED_DIR=runs/predictions/cmb_test_3000_random_noleak_v8_rl_only
METRIC_DIR=runs/metrics/cmb_test_3000_random_noleak_v8_rl_only
FULL_PRED_DIR=runs/predictions/cmb_test_full_best_v8_rl_only
FULL_METRIC_DIR=runs/metrics/cmb_test_full_best_v8_rl_only

mkdir -p runs/logs "$PRED_DIR" "$METRIC_DIR" "$FULL_PRED_DIR" "$FULL_METRIC_DIR"

LOCK=runs/logs/rl_only_hard_exact_v8.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[rl-only-v8] another runner is already active"
  exit 0
fi

log() {
  echo "[rl-only-v8] $* $(date '+%F %T')"
}

evaluate_adapter() {
  local name="$1"
  local adapter="$2"
  local pred="$PRED_DIR/${name}.jsonl"
  local metric="$METRIC_DIR/${name}.json"
  rm -f "$pred" "$metric"
  log "evaluating ${name}"
  /root/miniconda3/bin/python -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$adapter" \
    --eval "$SCREEN_EVAL" \
    --task choice \
    --out "$pred" \
    --start 0 \
    --limit 3000 \
    --max-new-tokens 16 \
    --progress-every 100 \
    --flush-every 1 \
    --resume
  /root/miniconda3/bin/python scripts/eval_choice_predictions_robust.py \
    --predictions "$pred" \
    --out "$metric"
}

sweep_adapter() {
  local tag="$1"
  local root="$2"
  local dirs=()
  while IFS= read -r d; do
    dirs+=("$d")
  done < <(find "$root" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  for d in "${dirs[@]}"; do
    local base
    base=$(basename "$d" | tr '-' '_')
    if [ -f "$d/adapter_config.json" ] || [ -f "$d/adapter_model.safetensors" ]; then
      evaluate_adapter "${tag}_${base}" "$d"
    fi
  done
  if [ -f "$root/adapter_config.json" ] || [ -f "$root/adapter_model.safetensors" ]; then
    evaluate_adapter "${tag}_final" "$root"
  fi
}

train_and_sweep() {
  local tag="$1"
  local config="$2"
  local root="$3"
  log "training ${tag} from fixed SFT adapter"
  if llamafactory-cli train "$config"; then
    sweep_adapter "$tag" "$root"
  else
    log "${tag} failed; continuing to next RL algorithm"
  fi
}

log "verifying original SFT adapter remains fixed at ${SFT_ADAPTER}"
test -f "$SFT_ADAPTER/adapter_config.json"

log "building RL-only hard-exact preference data from SFT wrong predictions"
/root/miniconda3/bin/python scripts/build_hard_exact_preference.py \
  --predictions "$SOURCE_PRED" \
  --exclude-eval data/eval/cmb_test_choice_eval.jsonl data/eval/cmb_test_choice_3000_random_noleak.jsonl \
  --out "$DATASET" \
  --summary "$DATA_SUMMARY" \
  --max-rows 6000 \
  --min-rows 1000

log "training and sweeping DPO"
train_and_sweep dpo \
  configs/llamafactory/qwen25_3b_choice_hard_exact_dpo_lora.yaml \
  /root/autodl-tmp/qwen-med-runs/general-med-choice-hard-exact-dpo-lora

log "training and sweeping IPO"
train_and_sweep ipo \
  configs/llamafactory/qwen25_3b_choice_hard_exact_ipo_lora.yaml \
  /root/autodl-tmp/qwen-med-runs/general-med-choice-hard-exact-ipo-lora

log "training and sweeping ORPO"
train_and_sweep orpo \
  configs/llamafactory/qwen25_3b_choice_hard_exact_orpo_lora.yaml \
  /root/autodl-tmp/qwen-med-runs/general-med-choice-hard-exact-orpo-lora

log "selecting best RL-only checkpoint"
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

metric_dir = Path("runs/metrics/cmb_test_3000_random_noleak_v8_rl_only")
baseline = json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v4/sft.json").read_text(encoding="utf-8"))["accuracy"]
items = {}
for p in sorted(metric_dir.glob("*.json")):
    if p.name in {"selection.json", "gate.json"}:
        continue
    m = json.loads(p.read_text(encoding="utf-8"))
    if "accuracy" not in m:
        continue
    items[p.stem] = {
        "accuracy": m["accuracy"],
        "correct": m.get("correct"),
        "total": m.get("total"),
        "single_accuracy": (m.get("single_choice") or {}).get("accuracy"),
        "multi_accuracy": (m.get("multi_choice") or {}).get("accuracy"),
        "invalid_prediction_count": m.get("invalid_prediction_count", m.get("invalid", 0)),
    }

def adapter_for(name: str) -> str:
    algo, _, suffix = name.partition("_")
    root_map = {
        "dpo": "/root/autodl-tmp/qwen-med-runs/general-med-choice-hard-exact-dpo-lora",
        "ipo": "/root/autodl-tmp/qwen-med-runs/general-med-choice-hard-exact-ipo-lora",
        "orpo": "/root/autodl-tmp/qwen-med-runs/general-med-choice-hard-exact-orpo-lora",
    }
    root = root_map[algo]
    if suffix == "final":
        return root
    if suffix.startswith("checkpoint_"):
        return f"{root}/{suffix.replace('_', '-')}"
    return root

best = max(items, key=lambda k: items[k]["accuracy"]) if items else None
summary = {
    "baseline_sft_accuracy": baseline,
    "models": items,
    "best_model": best,
    "best_accuracy": items[best]["accuracy"] if best else None,
    "best_adapter": adapter_for(best) if best else None,
    "delta_vs_sft_pp": round((items[best]["accuracy"] - baseline) * 100, 4) if best else None,
    "run_full": bool(best and items[best]["accuracy"] > baseline),
}
(metric_dir / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

RUN_FULL=$(/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
print("1" if json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v8_rl_only/selection.json").read_text(encoding="utf-8"))["run_full"] else "0")
PY
)

if [ "$RUN_FULL" = "1" ]; then
  BEST_ADAPTER=$(/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v8_rl_only/selection.json").read_text(encoding="utf-8"))["best_adapter"])
PY
)
  BEST_MODEL=$(/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v8_rl_only/selection.json").read_text(encoding="utf-8"))["best_model"])
PY
)
  log "best RL-only model beat SFT; running full CMB-test for ${BEST_MODEL}"
  /root/miniconda3/bin/python -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$BEST_ADAPTER" \
    --eval "$FULL_EVAL" \
    --task choice \
    --out "$FULL_PRED_DIR/${BEST_MODEL}.jsonl" \
    --start 0 \
    --limit 11200 \
    --max-new-tokens 16 \
    --progress-every 200 \
    --flush-every 1 \
    --resume
  /root/miniconda3/bin/python scripts/eval_choice_predictions_robust.py \
    --predictions "$FULL_PRED_DIR/${BEST_MODEL}.jsonl" \
    --out "$FULL_METRIC_DIR/${BEST_MODEL}.json"
else
  log "no RL-only checkpoint beat SFT on random-3000; not running full CMB-test"
fi

log "RL-only hard-exact v8 complete"
