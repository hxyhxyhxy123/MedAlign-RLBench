from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import yaml
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import dataset_info
from tqdm import tqdm


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def pick_dataset_name(spec: dict[str, Any]) -> list[str]:
    if "name" in spec:
        return [spec["name"]]
    return list(spec.get("candidates", []))


def try_load_rows(name: str, config: str | None, split_candidates: list[str], max_rows: int) -> tuple[str | None, list[dict[str, Any]], str | None]:
    last_error = None
    for split in split_candidates:
        try:
            ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
            rows = [dict(x) for x in itertools.islice(ds, max_rows)]
            return split, rows, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, [], last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--max-rows", type=int, default=8)
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    samples_dir = Path(manifest["paths"]["samples_dir"])
    metadata_dir = Path(manifest["paths"]["metadata_dir"])
    probe: dict[str, Any] = {"datasets": {}, "base_model": manifest.get("base_model")}

    for key, spec in tqdm(manifest["datasets"].items(), desc="probe datasets"):
        record: dict[str, Any] = {"purpose": spec.get("purpose"), "attempts": []}
        names = pick_dataset_name(spec)
        configs = spec.get("configs")
        split_candidates = spec.get("split_candidates", ["train"])
        max_rows = min(args.max_rows, int(spec.get("max_rows_no_gpu", args.max_rows)))

        for name in names:
            try:
                info = dataset_info(name)
                record["attempts"].append({"name": name, "info": "ok", "sha": getattr(info, "sha", None)})
            except Exception as exc:
                record["attempts"].append({"name": name, "info_error": f"{type(exc).__name__}: {exc}"})

            config_list = configs
            if config_list is None:
                try:
                    config_list = get_dataset_config_names(name, trust_remote_code=True)
                    if not config_list:
                        config_list = [None]
                except Exception:
                    config_list = [None]

            for config in config_list:
                split, rows, error = try_load_rows(name, config, split_candidates, max_rows)
                attempt = {"name": name, "config": config, "split": split, "rows": len(rows), "error": error}
                record["attempts"].append(attempt)
                if rows:
                    sample_name = clean_name(f"{key}_{name}_{config or 'default'}_{split}")
                    sample_path = samples_dir / f"{sample_name}.jsonl"
                    dump_jsonl(sample_path, rows)
                    record["selected"] = {
                        "name": name,
                        "config": config,
                        "split": split,
                        "sample_path": str(sample_path),
                        "columns": sorted(rows[0].keys()),
                        "rows": len(rows),
                    }
                    break
            if "selected" in record:
                break

        probe["datasets"][key] = record

    dump_json(metadata_dir / "dataset_probe.json", probe)
    print(json.dumps(probe, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
