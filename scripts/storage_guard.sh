#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

echo "[storage] filesystem usage"
df -h "$PROJECT_DIR" || df -h .

project_kb=$(du -sk "$PROJECT_DIR" 2>/dev/null | awk '{print $1}')
project_gb=$(awk "BEGIN {printf \"%.2f\", $project_kb/1024/1024}")
avail_kb=$(df -Pk "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
avail_gb=$(awk "BEGIN {printf \"%.2f\", $avail_kb/1024/1024}")

echo "[storage] project_size=${project_gb}GB"
echo "[storage] available=${avail_gb}GB"

too_low=$(awk "BEGIN {print ($avail_gb < $MIN_FREE_GB) ? 1 : 0}")
if [ "$too_low" = "1" ]; then
  echo "[storage] ERROR: available space is below ${MIN_FREE_GB}GB. Stop to avoid filling the disk."
  exit 2
fi
