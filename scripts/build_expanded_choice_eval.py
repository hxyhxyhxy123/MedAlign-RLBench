from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_prompt(text: Any) -> str:
    text = str(text or "")
    text = text.replace("请回答下面的中文医学考试题，并给出简短解析。\n", "")
    text = text.replace("请回答下面的中文医学考试题，并给出简短解析。", "")
    return re.sub(r"\s+", " ", text).strip()


def train_prompt_set(paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if "input" in row:
                seen.add(normalize_prompt(row.get("input", "")))
            if "prompt" in row:
                seen.add(normalize_prompt(row.get("prompt", "")))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/eval/cmb_test_choice_eval.jsonl")
    parser.add_argument("--out", default="data/eval/cmb_test_choice_500_noleak.jsonl")
    parser.add_argument("--summary", default="data/eval/cmb_test_choice_500_noleak_summary.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--train", nargs="*", default=[
        "data/llamafactory/stage1_med_sft_train.jsonl",
        "data/llamafactory/stage1_med_dpo_train.jsonl",
        "data/llamafactory/full_med_sft_train.jsonl",
        "data/llamafactory/full_med_dpo_train.jsonl",
    ])
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.source))
    seen = train_prompt_set([Path(x) for x in args.train])

    selected: list[dict[str, Any]] = []
    skipped_seen = 0
    answer_counts: dict[str, int] = {}
    qtype_counts: dict[str, int] = {}
    for row in source_rows:
        prompt = normalize_prompt(row.get("prompt", ""))
        if prompt in seen:
            skipped_seen += 1
            continue
        selected.append(row)
        answer = "".join(sorted(set(str(row.get("answer", "")).upper())))
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
        qtype = str(row.get("question_type", "unknown"))
        qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1
        if len(selected) >= args.limit:
            break

    write_jsonl(Path(args.out), selected)
    summary = {
        "source": args.source,
        "out": args.out,
        "requested_limit": args.limit,
        "source_rows": len(source_rows),
        "selected_rows": len(selected),
        "train_prompt_exact_seen": skipped_seen,
        "answer_counts": dict(sorted(answer_counts.items())),
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "train_files": args.train,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
