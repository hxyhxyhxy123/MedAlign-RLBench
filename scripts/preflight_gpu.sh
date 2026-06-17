#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/qwen-med-align-auto}"
PY_BIN="${PYTHON:-/root/miniconda3/bin/python}"

cd "$PROJECT_DIR"

echo "[preflight] checking project/data"
"$PY_BIN" scripts/check_project_ready.py

echo "[preflight] checking GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[preflight] no usable nvidia-smi yet; still in no-GPU mode"
fi

echo "[preflight] checking Python packages"
"$PY_BIN" - <<'PY'
mods = ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]
for name in mods:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "installed"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, exc)
PY
