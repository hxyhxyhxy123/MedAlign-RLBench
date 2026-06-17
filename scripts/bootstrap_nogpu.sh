#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/qwen-med-align-auto}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

mkdir -p "$PROJECT_DIR" "$HF_HOME" /root/autodl-tmp/qwen-med-data /root/autodl-tmp/qwen-med-runs

export HF_HOME
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PIP_NO_CACHE_DIR=1
export GIT_LFS_SKIP_SMUDGE=1

cd "$PROJECT_DIR"

echo "[bootstrap] project=$PROJECT_DIR"
echo "[bootstrap] HF_HOME=$HF_HOME"

PY_BIN="${PYTHON:-}"
if [ -z "$PY_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
  elif [ -x /root/miniconda3/bin/python ]; then
    PY_BIN="/root/miniconda3/bin/python"
  else
    echo "[bootstrap] ERROR: no Python executable found"
    exit 1
  fi
fi

echo "[bootstrap] python=$PY_BIN"
"$PY_BIN" -V

"$PY_BIN" -m pip install -U pip setuptools wheel
"$PY_BIN" -m pip install --no-cache-dir -r requirements-nogpu.txt

bash scripts/storage_guard.sh
bash scripts/clone_baselines.sh
"$PY_BIN" scripts/build_seed_data.py --manifest configs/data_manifest.yaml
"$PY_BIN" scripts/build_stage1_data.py --out data/stage1 --sft-size 30000 --dpo-size 25000
if [ "${RUN_HF_PROBE:-0}" = "1" ]; then
  "$PY_BIN" scripts/inspect_datasets.py --manifest configs/data_manifest.yaml --max-rows 4
else
  echo "[bootstrap] skip HF dataset probe in no-GPU mode; set RUN_HF_PROBE=1 to enable it"
fi
bash scripts/storage_guard.sh

echo "[bootstrap] no-GPU preparation finished"
