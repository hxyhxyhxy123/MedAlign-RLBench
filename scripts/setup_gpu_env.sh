#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/qwen-med-align-auto}"
PY_BIN="${PYTHON:-/root/miniconda3/bin/python}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PIP_NO_CACHE_DIR=1

cd "$PROJECT_DIR"

echo "[setup] project=$PROJECT_DIR"
echo "[setup] python=$PY_BIN"
"$PY_BIN" -V

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[setup] nvidia-smi not found. This script should be run after GPU boot."
fi

"$PY_BIN" -m pip install -U pip setuptools wheel

# LLaMA-Factory core dependency ranges from its pyproject.
"$PY_BIN" -m pip install --no-cache-dir \
  "transformers>=4.55.0,<=5.6.0,!=4.52.0,!=4.57.0" \
  "datasets>=2.16.0,<=4.0.0" \
  "accelerate>=1.3.0,<=1.11.0" \
  "peft>=0.18.0,<=0.18.1" \
  "trl>=0.18.0,<=0.24.0" \
  "torchdata>=0.10.0,<=0.11.0" \
  sentencepiece tiktoken safetensors einops pandas scipy protobuf pyyaml fire omegaconf tyro \
  "bitsandbytes>=0.45.0" \
  "deepspeed>=0.10.0,<=0.18.4"

"$PY_BIN" -m pip install --no-cache-dir -e baselines/LLaMA-Factory --no-deps

"$PY_BIN" scripts/check_gpu_env.py
"$PY_BIN" scripts/check_project_ready.py

echo "[setup] GPU training environment check finished."
