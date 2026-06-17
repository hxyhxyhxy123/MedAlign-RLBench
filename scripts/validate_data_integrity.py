from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def read_jsonl_stats(path: Path, required: list[str], ranking: bool = False, optional_empty: set[str] | None = None) -> dict[str, Any]:
    optional_empty = optional_empty or set()
    stats: dict[str, Any] = {
        "file": str(path),
        "exists": path.exists(),
        "rows": 0,
        "bad_json": 0,
        "missing_required": 0,
        "empty_required": 0,
        "chosen_equals_rejected": 0,
        "duplicate_prompt_rows": 0,
        "source_counts": {},
        "max_field_chars": {},
        "sample": None,
    }
    if not path.exists():
        return stats
    prompts = Counter()
    source_counts = Counter()
    max_chars = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            stats["rows"] += 1
            if stats["sample"] is None:
                stats["sample"] = {k: str(row.get(k, ""))[:180] for k in required}
            for key in required:
                if key not in row:
                    stats["missing_required"] += 1
                elif key not in optional_empty and str(row.get(key, "")).strip() == "":
                    stats["empty_required"] += 1
                max_chars[key] = max(max_chars[key], len(str(row.get(key, ""))))
            prompt_key = "prompt" if "prompt" in row else "instruction"
            prompt = re.sub(r"\s+", " ", str(row.get(prompt_key, ""))).strip()
            if prompt:
                prompts[prompt] += 1
            if ranking and str(row.get("chosen", "")) == str(row.get("rejected", "")):
                stats["chosen_equals_rejected"] += 1
            source_counts[str(row.get("source", "unknown"))] += 1
    stats["duplicate_prompt_rows"] = sum(v - 1 for v in prompts.values() if v > 1)
    stats["source_counts"] = dict(source_counts)
    stats["max_field_chars"] = dict(max_chars)
    return stats


def parse_config_datasets(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = data.get("dataset", "")
    return [x.strip() for x in str(value).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/llamafactory")
    parser.add_argument("--configs-dir", default="configs/llamafactory")
    parser.add_argument("--out-json", default="data/metadata/data_integrity_report.json")
    parser.add_argument("--out-md", default="docs/data_integrity_report.md")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))

    datasets: dict[str, Any] = {}
    problems = []
    for name, spec in sorted(info.items()):
        file_name = spec.get("file_name")
        if not file_name:
            continue
        cols = spec.get("columns", {})
        ranking = bool(spec.get("ranking"))
        optional_empty: set[str] = set()
        if ranking:
            required = [cols.get("prompt", "prompt"), cols.get("chosen", "chosen"), cols.get("rejected", "rejected")]
        elif set(cols.keys()) == {"prompt"}:
            required = [cols.get("prompt", "text")]
        else:
            required = [cols.get("prompt", "instruction"), cols.get("query", "input"), cols.get("response", "output")]
            optional_empty.add(cols.get("query", "input"))
        stats = read_jsonl_stats(dataset_dir / file_name, required, ranking=ranking, optional_empty=optional_empty)
        datasets[name] = stats
        if not stats["exists"] or stats["rows"] == 0 or stats["bad_json"] or stats["missing_required"] or stats["empty_required"]:
            problems.append(name)
        if ranking and stats["chosen_equals_rejected"]:
            problems.append(name)

    config_refs = {}
    missing_refs = []
    for cfg in sorted(Path(args.configs_dir).glob("*.yaml")):
        refs = parse_config_datasets(cfg)
        config_refs[cfg.name] = refs
        for ref in refs:
            if ref not in info:
                missing_refs.append({"config": cfg.name, "dataset": ref})

    eval_files = {}
    for path in sorted(Path("data/eval").glob("*.jsonl")):
        required = ["prompt", "answer"] if "redflag" not in path.name else ["prompt", "chosen", "rejected"]
        eval_files[path.name] = read_jsonl_stats(path, required, ranking="redflag" in path.name)

    report = {
        "dataset_dir": str(dataset_dir),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "config_refs": config_refs,
        "missing_config_refs": missing_refs,
        "problem_datasets": sorted(set(problems)),
        "eval_files": eval_files,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Data Integrity Report",
        "",
        f"- Dataset registry: `{info_path}`",
        f"- Registered datasets: {len(datasets)}",
        f"- Problem datasets: {len(set(problems))}",
        f"- Missing config refs: {len(missing_refs)}",
        "",
        "## LLaMA-Factory Datasets",
        "",
        "| name | rows | bad_json | empty_required | chosen=rejected | file |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, stats in datasets.items():
        lines.append(
            f"| {name} | {stats['rows']} | {stats['bad_json']} | {stats['empty_required']} | {stats['chosen_equals_rejected']} | {Path(stats['file']).name} |"
        )
    lines.extend(["", "## Eval Files", "", "| file | rows | bad_json | empty_required |", "|---|---:|---:|---:|"])
    for name, stats in eval_files.items():
        lines.append(f"| {name} | {stats['rows']} | {stats['bad_json']} | {stats['empty_required']} |")
    if missing_refs:
        lines.extend(["", "## Missing Config Refs", ""])
        for item in missing_refs:
            lines.append(f"- {item['config']}: {item['dataset']}")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"problem_datasets": sorted(set(problems)), "missing_config_refs": missing_refs}, ensure_ascii=False, indent=2))
    if problems or missing_refs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
