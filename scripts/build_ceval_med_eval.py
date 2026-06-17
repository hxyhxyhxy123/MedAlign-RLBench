from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIGS = ["basic_medicine", "clinical_medicine", "physician"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row: dict[str, Any], subject: str) -> str:
    options = []
    for key in ["A", "B", "C", "D"]:
        value = str(row.get(key, "")).strip()
        if value:
            options.append(f"{key}. {value}")
    question = str(row.get("question", "")).strip()
    option_text = "\n".join(options)
    return (
        "Question source: C-Eval medical / "
        f"{subject}\n{question}\n{option_text}\n"
        "Please output only the final answer option, such as A, B, C, or D."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval/ceval_med_val_eval.jsonl")
    parser.add_argument("--summary-out", default="data/eval/ceval_med_val_summary.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--split", default="val")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    by_subject: dict[str, int] = {}
    per_subject_limit = max(1, args.limit // max(1, len(args.configs)))
    remainder = args.limit - per_subject_limit * len(args.configs)

    for idx, config in enumerate(args.configs):
        subject_limit = per_subject_limit + (1 if idx < remainder else 0)
        ds = load_dataset("ceval/ceval-exam", config, split=args.split, streaming=True)
        count = 0
        for raw in ds:
            row = dict(raw)
            answer = str(row.get("answer", "")).strip().upper()
            if not answer:
                continue
            rows.append(
                {
                    "id": f"ceval-{config}-{row.get('id', count)}",
                    "prompt": build_prompt(row, config),
                    "answer": answer,
                    "explanation": str(row.get("explanation", "")),
                    "source": "C-Eval-med",
                    "subject": config,
                }
            )
            count += 1
            if count >= subject_limit:
                break
        by_subject[config] = count

    rows = rows[: args.limit]
    write_jsonl(Path(args.out), rows)
    summary = {
        "dataset": "ceval/ceval-exam",
        "split": args.split,
        "total": len(rows),
        "by_subject": by_subject,
        "output": args.out,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
