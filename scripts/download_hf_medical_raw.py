from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


TARGETS = [
    {
        "repo_id": "FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K",
        "files": ["HuatuoGPT2-GPT4-SFT-140K.json", "README.md"],
    },
    {
        "repo_id": "FreedomIntelligence/HuatuoGPT-sft-data-v1",
        "files": ["HuatuoGPT_sft_data_v1.jsonl", "README.md"],
    },
    {
        "repo_id": "FreedomIntelligence/Huatuo26M-Lite",
        "files": ["format_data.jsonl", "README.md"],
    },
    {
        "repo_id": "Flmc/DISC-Med-SFT",
        "files": ["DISC-Med-SFT_released.jsonl", "README.md"],
    },
    {
        "repo_id": "shibing624/medical",
        "files": [
            "finetune/train_zh_0.json",
            "finetune/valid_zh_0.json",
            "finetune/test_zh_0.json",
            "pretrain/medical_book_zh.json",
            "pretrain/train_encyclopedia.json",
            "pretrain/valid_encyclopedia.json",
            "pretrain/test_encyclopedia.json",
            "reward/train.json",
            "reward/valid.json",
            "reward/test.json",
            "README.md",
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/hf")
    parser.add_argument("--manifest", default="data/metadata/hf_download_manifest.json")
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
                    resume_download=True,
                )
                p = Path(path)
                rec.update({"ok": True, "path": str(p), "size": p.stat().st_size})
                print(json.dumps(rec, ensure_ascii=False), flush=True)
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
