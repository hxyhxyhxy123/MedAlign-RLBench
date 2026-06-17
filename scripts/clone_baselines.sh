#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-baselines}"
mkdir -p "$BASE_DIR"

export GIT_LFS_SKIP_SMUDGE=1
git config --global http.version HTTP/1.1 || true
git config --global http.postBuffer 524288000 || true

clone_or_update() {
  local url="$1"
  local dir="$2"
  local required="${3:-optional}"
  if [ -d "$BASE_DIR/$dir/.git" ]; then
    echo "[baseline] exists: $dir"
    git -C "$BASE_DIR/$dir" remote -v | head -n 1 || true
  else
    echo "[baseline] clone: $dir <- $url"
    local ok=0
    for attempt in 1 2 3; do
      echo "[baseline] attempt $attempt for $dir"
      if git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --filter=blob:none "$url" "$BASE_DIR/$dir"; then
        ok=1
        break
      fi
      rm -rf "$BASE_DIR/$dir"
      sleep $((attempt * 3))
    done
    if [ "$ok" != "1" ]; then
      echo "[baseline] retry without partial clone filter for $dir"
      if git -c http.version=HTTP/1.1 clone --depth 1 --single-branch "$url" "$BASE_DIR/$dir"; then
        ok=1
      fi
    fi
    if [ "$ok" != "1" ]; then
      echo "[baseline] FAILED: $dir"
      echo "$dir $url" >> "$BASE_DIR/clone_failures.txt"
      if [ "$required" = "required" ]; then
        return 1
      fi
      return 0
    fi
  fi
  du -sh "$BASE_DIR/$dir" || true
}

clone_or_update https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory required
clone_or_update https://github.com/shibing624/MedicalGPT.git MedicalGPT optional
clone_or_update https://github.com/FreedomIntelligence/HuatuoGPT-II.git HuatuoGPT-II optional
clone_or_update https://github.com/FreedomIntelligence/CMB.git CMB optional
clone_or_update https://github.com/FudanDISC/DISC-MedLLM.git DISC-MedLLM optional
clone_or_update https://github.com/huggingface/trl.git trl optional
clone_or_update https://github.com/vllm-project/vllm.git vllm optional

cat > "$BASE_DIR/BASELINE_MANIFEST.md" <<'EOF'
# Baseline Manifest

Core training baseline:

- LLaMA-Factory: Qwen2.5 SFT/LoRA/QLoRA/DPO command-line training.

Medical references:

- MedicalGPT: Chinese medical LLM FT/SFT/DPO/GRPO training workflow reference.
- HuatuoGPT-II: medical SFT and model adaptation reference.
- CMB: Chinese Medical Benchmark reference.
- DISC-MedLLM: Chinese medical dialogue model/data construction reference.

Alignment/deployment references:

- TRL: SFT/DPO/GRPO trainer correctness reference.
- vLLM: deployment, serving, KV cache/PagedAttention benchmark reference.

Large model weights and full datasets are intentionally not downloaded in no-GPU mode.
EOF

du -sh "$BASE_DIR" || true
