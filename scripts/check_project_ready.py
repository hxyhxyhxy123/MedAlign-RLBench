from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


REQUIRED_BASELINES = [
    "LLaMA-Factory",
    "MedicalGPT",
    "HuatuoGPT-II",
    "CMB",
    "DISC-MedLLM",
    "trl",
    "vllm",
]

REQUIRED_DATASETS = [
    "stage1_med_sft_train",
    "stage1_med_dpo_train",
    "stage1_med_mpo_train",
    "full_med_sft_train",
    "full_med_dpo_train",
    "full_med_mpo_train",
    "hf_med_sft_full",
    "hf_med_dpo_full",
    "extra_med_sft_full",
    "extra_med_dpo_full",
    "extra_med_pretrain_full",
]

REQUIRED_CONFIGS = [
    "qwen25_3b_sft_lora.yaml",
    "qwen25_3b_sft_lora_zero2.yaml",
    "qwen25_3b_sft_qlora.yaml",
    "qwen25_3b_dpo_lora.yaml",
    "qwen25_3b_dpo_lora_zero2.yaml",
    "qwen25_3b_sft_lora_complete.yaml",
    "qwen25_3b_sft_qlora_complete.yaml",
    "qwen25_3b_dpo_lora_complete.yaml",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_fast(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def parse_dataset_refs(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    value = cfg.get("dataset", "")
    return [x.strip() for x in str(value).split(",") if x.strip()]


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    except Exception as exc:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--deep-count", action="store_true", help="Count rows for the largest jsonl files.")
    parser.add_argument("--out", default="data/metadata/project_ready_report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"root": str(root)}

    baselines = {}
    for name in REQUIRED_BASELINES:
        path = root / "baselines" / name
        baselines[name] = {"exists": path.exists(), "size": None}
        if not path.exists():
            errors.append(f"missing baseline: {name}")
    report["baselines"] = baselines

    dataset_info_path = root / "data" / "llamafactory" / "dataset_info.json"
    if not dataset_info_path.exists():
        errors.append("missing data/llamafactory/dataset_info.json")
        dataset_info = {}
    else:
        dataset_info = read_json(dataset_info_path)
    report["dataset_info_count"] = len(dataset_info)

    dataset_checks = {}
    for name in REQUIRED_DATASETS:
        spec = dataset_info.get(name)
        if not spec:
            errors.append(f"dataset not registered: {name}")
            continue
        path = root / "data" / "llamafactory" / spec["file_name"]
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows = None
        if exists and (args.deep_count or size < 250_000_000):
            rows = count_jsonl_fast(path)
        dataset_checks[name] = {"exists": exists, "file": str(path), "size": size, "rows": rows}
        if not exists or size == 0:
            errors.append(f"dataset file missing/empty: {name}")
    report["dataset_checks"] = dataset_checks

    config_checks = {}
    for cfg_name in REQUIRED_CONFIGS:
        path = root / "configs" / "llamafactory" / cfg_name
        if not path.exists():
            errors.append(f"missing config: {cfg_name}")
            continue
        refs = parse_dataset_refs(path)
        missing = [x for x in refs if x not in dataset_info]
        config_checks[cfg_name] = {"refs": refs, "missing_refs": missing}
        for ref in missing:
            errors.append(f"{cfg_name} references missing dataset {ref}")
    report["config_checks"] = config_checks

    integrity_path = root / "data" / "metadata" / "data_integrity_report.json"
    if integrity_path.exists():
        integrity = read_json(integrity_path)
        report["integrity_problem_datasets"] = integrity.get("problem_datasets", [])
        report["integrity_missing_config_refs"] = integrity.get("missing_config_refs", [])
        if integrity.get("problem_datasets"):
            errors.append("data integrity report has problem datasets")
        if integrity.get("missing_config_refs"):
            errors.append("data integrity report has missing config refs")
    else:
        warnings.append("data integrity report json missing")

    py = shutil.which("python3") or shutil.which("python") or "/root/miniconda3/bin/python"
    report["python"] = run([py, "-V"])
    report["git"] = run(["git", "--version"])
    report["llamafactory_source_import"] = run(
        [py, "-c", "import sys; sys.path.insert(0,'baselines/LLaMA-Factory/src'); import llamafactory; print('ok')"],
        cwd=root,
        timeout=60,
    )
    report["llamafactory_cli"] = run(["bash", "-lc", "command -v llamafactory-cli || true"], cwd=root)
    report["nvidia_smi"] = run(["bash", "-lc", "command -v nvidia-smi >/dev/null && nvidia-smi -L || echo no-gpu"], cwd=root)

    report["warnings"] = warnings
    report["errors"] = errors
    report["ready_no_gpu"] = not errors
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ready_no_gpu": report["ready_no_gpu"], "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
