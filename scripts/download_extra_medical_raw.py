from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


TARGETS = [
    {
        "repo_id": "Mxode/Chinese-Medical-Instruct-1M",
        "files": ["medical-train.jsonl", "README.md"],
    },
    {
        "repo_id": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "files": [
            "medical_o1_sft_Chinese.json",
            "medical_o1_sft_mix_Chinese.json",
            "README.md",
        ],
    },
    {
        "repo_id": "fzkuji/CMExam",
        "files": ["train.json", "valid.json", "test.json", "README.md"],
    },
    {
        "repo_id": "lavita/ChatDoctor-HealthCareMagic-100k",
        "files": ["data/train-00000-of-00001-5e7cb295b9cff0bf.parquet", "README.md"],
    },
    {
        "repo_id": "medalpaca/medical_meadow_medical_flashcards",
        "files": ["data/train-00000-of-00001.parquet", "README.md"],
    },
    {
        "repo_id": "ruslanmv/ai-medical-dataset",
        "files": [f"data/train-{i:05d}-of-00018.parquet" for i in range(18)] + ["README.md"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/hf_extra")
    parser.add_argument("--manifest", default="data/metadata/hf_extra_download_manifest.json")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    out = Path(args.out)
    records = []
    for target in TARGETS:
        repo_id = target["repo_id"]
        local_dir = out / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        for filename in target["files"]:
            rec = {"repo_id": repo_id, "filename": filename, "ok": False}
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    endpoint=os.environ.get("HF_ENDPOINT"),
                    local_dir=str(local_dir),
                )
                p = Path(path)
                rec.update({"ok": True, "path": str(p), "size": p.stat().st_size})
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            records.append(rec)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [r for r in records if not r["ok"]]
    print(json.dumps({"downloaded": len(records) - len(failed), "failed": len(failed)}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
