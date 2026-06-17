#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-med-align-auto

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
ADAPTER=/root/autodl-tmp/qwen-med-runs/general-med-choice-answer-dpo-lora
SCREEN_EVAL=data/eval/cmb_test_choice_3000_random_noleak.jsonl
FULL_EVAL=data/eval/cmb_test_choice_eval.jsonl
PRED_DIR=runs/predictions/cmb_test_3000_random_noleak_v6
METRIC_DIR=runs/metrics/cmb_test_3000_random_noleak_v6
FULL_PRED_DIR=runs/predictions/cmb_test_full_best_v6
FULL_METRIC_DIR=runs/metrics/cmb_test_full_best_v6

mkdir -p runs/logs "$PRED_DIR" "$METRIC_DIR" "$FULL_PRED_DIR" "$FULL_METRIC_DIR"

LOCK=runs/logs/choice_answer_dpo_v6.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[choice-dpo-v6] another runner is already active"
  exit 0
fi

log() {
  echo "[choice-dpo-v6] $* $(date '+%F %T')"
}

log "building answer-only preference data"
/root/miniconda3/bin/python scripts/build_answer_only_preference.py \
  --sources data/eval/extra_cmexam_eval.jsonl data/llamafactory/full_med_sft_train.jsonl \
  --exclude-eval data/eval/cmb_test_choice_eval.jsonl data/eval/cmb_test_choice_3000_random_noleak.jsonl \
  --predictions runs/predictions/hard_choice_candidate_cmexam_12000_sft.jsonl \
  --out data/llamafactory/choice_answer_only_dpo.jsonl \
  --summary data/metadata/choice_answer_only_dpo_summary.json \
  --max-rows 60000 \
  --pairs-per-question 2 \
  --multi-repeat 3

log "training answer-only DPO"
llamafactory-cli train configs/llamafactory/qwen25_3b_choice_answer_dpo_lora.yaml

log "evaluating answer-only DPO on random-3000"
/root/miniconda3/bin/python -u scripts/generate_eval_predictions_sharded.py \
  --base-model "$BASE" \
  --adapter "$ADAPTER" \
  --eval "$SCREEN_EVAL" \
  --task choice \
  --out "$PRED_DIR/choice_answer_dpo.jsonl" \
  --start 0 \
  --limit 3000 \
  --max-new-tokens 16 \
  --progress-every 100 \
  --flush-every 1 \
  --resume
/root/miniconda3/bin/python scripts/eval_choice_predictions_robust.py \
  --predictions "$PRED_DIR/choice_answer_dpo.jsonl" \
  --out "$METRIC_DIR/choice_answer_dpo.json"

log "comparing with v4 best"
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
new = json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v6/choice_answer_dpo.json").read_text(encoding="utf-8"))
old = json.loads(Path("runs/metrics/cmb_test_3000_random_noleak_v4/sft.json").read_text(encoding="utf-8"))
summary = {
    "new_model": "choice_answer_dpo",
    "new_accuracy": new["accuracy"],
    "old_sft_accuracy": old["accuracy"],
    "delta_pp": round((new["accuracy"] - old["accuracy"]) * 100, 4),
    "run_full": new["accuracy"] > old["accuracy"],
}
Path("runs/metrics/cmb_test_3000_random_noleak_v6/selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

RUN_FULL=$(/root/miniconda3/bin/python - <<'PY'
import json
print("1" if json.load(open("runs/metrics/cmb_test_3000_random_noleak_v6/selection.json", encoding="utf-8"))["run_full"] else "0")
PY
)

if [ "$RUN_FULL" = "1" ]; then
  log "answer-only DPO beat SFT; running full CMB-test"
  /root/miniconda3/bin/python -u scripts/generate_eval_predictions_sharded.py \
    --base-model "$BASE" \
    --adapter "$ADAPTER" \
    --eval "$FULL_EVAL" \
    --task choice \
    --out "$FULL_PRED_DIR/choice_answer_dpo.jsonl" \
    --start 0 \
    --limit 11200 \
    --max-new-tokens 16 \
    --progress-every 200 \
    --flush-every 1 \
    --resume
  /root/miniconda3/bin/python scripts/eval_choice_predictions_robust.py \
    --predictions "$FULL_PRED_DIR/choice_answer_dpo.jsonl" \
    --out "$FULL_METRIC_DIR/choice_answer_dpo.json"
else
  log "answer-only DPO did not beat SFT on random-3000; skipping full CMB-test"
fi

log "choice answer DPO v6 complete"
