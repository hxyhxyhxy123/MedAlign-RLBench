#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-med-align-auto

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
SFT=/root/autodl-tmp/qwen-med-runs/general-med-lora
RS_ORPO=/root/autodl-tmp/qwen-med-runs/general-med-rs-orpo-lora
RS_DPO=/root/autodl-tmp/qwen-med-runs/general-med-rs-dpo-lora
SCREEN_EVAL=data/eval/cmb_test_choice_3000_random_noleak.jsonl
FULL_EVAL=data/eval/cmb_test_choice_eval.jsonl
SCREEN_PRED_DIR=runs/predictions/cmb_test_3000_random_noleak_v4
SCREEN_METRIC_DIR=runs/metrics/cmb_test_3000_random_noleak_v4
FULL_PRED_DIR=runs/predictions/cmb_test_full_best_v5
FULL_METRIC_DIR=runs/metrics/cmb_test_full_best_v5
SAMPLES=runs/predictions/rs_hard_cmexam_sft_samples_k6.jsonl
RS_PREF=data/llamafactory/rs_hard_choice_sft_errors.jsonl
RS_SUMMARY=data/metadata/rs_hard_choice_sft_errors_summary.json

mkdir -p runs/logs "$SCREEN_PRED_DIR" "$SCREEN_METRIC_DIR" "$FULL_PRED_DIR" "$FULL_METRIC_DIR"

LOCK=runs/logs/rs_orpo_repair_v5.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[rs-v5] another rs repair runner is already active"
  exit 0
fi

log() {
  echo "[rs-v5] $* $(date '+%F %T')"
}

wait_for_existing_full_eval() {
  log "waiting for existing full CMB-test eval to finish"
  while pgrep -f 'generate_eval_predictions_sharded.py.*cmb_test_full_best_v4' >/dev/null; do
    sleep 60
  done
  if [ -f runs/predictions/cmb_test_full_best_v4/sft.jsonl ] && [ ! -f runs/metrics/cmb_test_full_best_v4/sft.json ]; then
    python scripts/eval_choice_predictions_robust.py \
      --predictions runs/predictions/cmb_test_full_best_v4/sft.jsonl \
      --out runs/metrics/cmb_test_full_best_v4/sft.json
  fi
}

summarize_screen() {
  python - <<'PY'
import json
from pathlib import Path

metric_dir = Path("runs/metrics/cmb_test_3000_random_noleak_v4")
names = ["base", "sft", "dpo", "mpo", "hard_dpo", "hard_ipo", "rs_orpo", "rs_dpo"]
summary = {}
for name in names:
    p = metric_dir / f"{name}.json"
    if p.exists():
        summary[name] = json.loads(p.read_text(encoding="utf-8"))

adapter = {
    "base": "",
    "sft": "/root/autodl-tmp/qwen-med-runs/general-med-lora",
    "dpo": "/root/autodl-tmp/qwen-med-runs/general-med-dpo-lora",
    "mpo": "/root/autodl-tmp/qwen-med-runs/general-med-mpo-lora",
    "hard_dpo": "/root/autodl-tmp/qwen-med-runs/general-med-hard-dpo-lora",
    "hard_ipo": "/root/autodl-tmp/qwen-med-runs/general-med-hard-ipo-lora",
    "rs_orpo": "/root/autodl-tmp/qwen-med-runs/general-med-rs-orpo-lora",
    "rs_dpo": "/root/autodl-tmp/qwen-med-runs/general-med-rs-dpo-lora",
}

def rank_key(item):
    name, metrics = item
    acc = float(metrics.get("accuracy", 0.0))
    simplicity = {"sft": 2, "dpo": 1, "base": 0}.get(name, 0)
    return (round(acc, 6), simplicity)

best_name, best_metrics = max(summary.items(), key=rank_key)
out = {
    "summary": summary,
    "best_name": best_name,
    "best_accuracy": best_metrics.get("accuracy"),
    "best_adapter": adapter[best_name],
}
(metric_dir / "final_selection_v5.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

run_screen_eval() {
  local name=$1
  local adapter=$2
  python -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$adapter" \
    --eval "$SCREEN_EVAL" \
    --task choice \
    --out "$SCREEN_PRED_DIR/$name.jsonl" \
    --start 0 \
    --limit 3000 \
    --max-new-tokens 16 \
    --progress-every 100 \
    --flush-every 1 \
    --resume
  python scripts/eval_choice_predictions_robust.py \
    --predictions "$SCREEN_PRED_DIR/$name.jsonl" \
    --out "$SCREEN_METRIC_DIR/$name.json"
}

run_full_eval_for_best() {
  local best_name
  local best_adapter
  best_name=$(python - <<'PY'
import json
print(json.load(open("runs/metrics/cmb_test_3000_random_noleak_v4/final_selection_v5.json", encoding="utf-8"))["best_name"])
PY
)
  best_adapter=$(python - <<'PY'
import json
print(json.load(open("runs/metrics/cmb_test_3000_random_noleak_v4/final_selection_v5.json", encoding="utf-8"))["best_adapter"])
PY
)
  log "best on random-3000 is ${best_name}; running full CMB-test only if needed"
  if [ "$best_name" = "sft" ] && [ -f runs/metrics/cmb_test_full_best_v4/sft.json ]; then
    cp runs/metrics/cmb_test_full_best_v4/sft.json "$FULL_METRIC_DIR/sft.json"
    cp runs/predictions/cmb_test_full_best_v4/sft.jsonl "$FULL_PRED_DIR/sft.jsonl"
    return
  fi
  python -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$best_adapter" \
    --eval "$FULL_EVAL" \
    --task choice \
    --out "$FULL_PRED_DIR/$best_name.jsonl" \
    --start 0 \
    --limit 11200 \
    --max-new-tokens 16 \
    --progress-every 200 \
    --flush-every 1 \
    --resume
  python scripts/eval_choice_predictions_robust.py \
    --predictions "$FULL_PRED_DIR/$best_name.jsonl" \
    --out "$FULL_METRIC_DIR/$best_name.json"
}

wait_for_existing_full_eval

log "sampling multiple responses on hard CMExam prompts"
python -u scripts/generate_choice_rejection_samples.py \
  --base-model "$BASE" \
  --adapter "$SFT" \
  --predictions runs/predictions/hard_choice_candidate_cmexam_12000_sft.jsonl \
  --out "$SAMPLES" \
  --selection-mode wrong_or_multi \
  --max-prompts 5000 \
  --samples-per-prompt 6 \
  --max-new-tokens 32 \
  --temperature 0.8 \
  --top-p 0.9 \
  --top-k 40 \
  --progress-every 50 \
  --flush-every 1 \
  --resume

log "building reward-ranked rejection-sampling preference pairs"
python scripts/build_rs_preference_from_samples.py \
  --samples "$SAMPLES" \
  --out "$RS_PREF" \
  --summary "$RS_SUMMARY" \
  --min-rows 1500 \
  --max-rows 6000 \
  --min-margin 0.15 \
  --allow-oracle-chosen

log "training RS-ORPO"
llamafactory-cli train configs/llamafactory/qwen25_3b_rs_orpo_lora.yaml
log "evaluating RS-ORPO on random-3000"
run_screen_eval rs_orpo "$RS_ORPO"

log "training RS-DPO"
llamafactory-cli train configs/llamafactory/qwen25_3b_rs_dpo_lora.yaml
log "evaluating RS-DPO on random-3000"
run_screen_eval rs_dpo "$RS_DPO"

log "selecting best model on random-3000"
summarize_screen
run_full_eval_for_best
log "RS repair pipeline complete"
