from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ANSWER_PREFIX = "\u7b54\u6848\uff1a"
CHOICE_INSTRUCTION = (
    "\u8bf7\u53ea\u8f93\u51fa\u6700\u7ec8\u7b54\u6848\u9009\u9879"
    "\uff08\u4f8b\u5982 A \u6216 BCDEF\uff09\uff0c\u4e0d\u8981\u8f93\u51fa\u89e3\u6790\u3002"
)
COMMON_PREFIXES = [
    "\u8bf7\u56de\u7b54\u4e0b\u9762\u7684\u4e2d\u6587\u533b\u5b66\u8003\u8bd5\u9898\uff0c\u5e76\u7ed9\u51fa\u7b80\u77ed\u89e3\u6790\u3002",
    "\u8bf7\u53ea\u8f93\u51fa\u6700\u7ec8\u7b54\u6848\u9009\u9879\uff08\u4f8b\u5982 A \u6216 BCDEF\uff09\uff0c\u4e0d\u8981\u8f93\u51fa\u89e3\u6790\u3002",
    "\u8bf7\u53ea\u8f93\u51fa\u6700\u7ec8\u7b54\u6848\u9009\u9879\uff0c\u4f8b\u5982 A \u6216 BCDEF\uff0c\u4e0d\u8981\u8f93\u51fa\u89e3\u6790\u3002",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value or "").upper()))))


def extract_choice(text: Any) -> str:
    upper = str(text or "").upper()
    patterns = [
        r"(?:\u6b63\u786e\u7b54\u6848|\u7b54\u6848|\u6700\u7ec8\u9009\u9879|\u9009\u62e9|ANSWER|CHOICE)\s*[:\uff1a\u4e3a\u662f-]*\s*([A-F](?:\s*[\u3001,\uff0c ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[\u3001,\uff0c ]?\s*[A-F]){0,5})\s*(?:$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            choice = normalize_choice(match.group(1))
            if choice:
                return choice
    compact = re.sub(r"\s+", "", upper)
    match = re.search(r"(?<![A-Z])([A-F]{1,6})(?![A-Z])", compact)
    return normalize_choice(match.group(1)) if match else ""


def strip_instruction(prompt: str) -> str:
    text = str(prompt or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in COMMON_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text


def prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", "", strip_instruction(prompt))


def load_exclude_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            prompt = str(row.get("prompt") or row.get("input") or row.get("question") or "")
            if prompt:
                keys.add(prompt_key(prompt))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--exclude-eval", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--answer-only", action="store_true", default=True)
    args = parser.parse_args()

    exclude_keys = load_exclude_keys([Path(p) for p in args.exclude_eval])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: dict[str, Any] = {
        "predictions": args.predictions,
        "exclude_eval": args.exclude_eval,
        "exclude_key_count": len(exclude_keys),
        "prediction_rows": 0,
        "usable_rows": 0,
        "correct_rows": 0,
        "wrong_rows": 0,
        "invalid_prediction_rows": 0,
        "skipped_eval_overlap": 0,
        "skipped_duplicate": 0,
        "skipped_no_gold": 0,
        "single_wrong_rows": 0,
        "multi_wrong_rows": 0,
        "by_question_type": {},
    }

    for row in iter_jsonl(Path(args.predictions)):
        stats["prediction_rows"] += 1
        prompt = str(row.get("prompt") or row.get("input") or row.get("question") or "")
        gold = normalize_choice(row.get("answer") or row.get("gold_answer") or row.get("gold_choice") or "")
        if not prompt or not gold:
            stats["skipped_no_gold"] += 1
            continue
        key = prompt_key(prompt)
        if key in exclude_keys:
            stats["skipped_eval_overlap"] += 1
            continue
        if key in seen:
            stats["skipped_duplicate"] += 1
            continue
        pred = extract_choice(row.get("prediction") or row.get("output") or row.get("model_answer") or "")
        stats["usable_rows"] += 1
        stats["invalid_prediction_rows"] += int(not pred)
        stats["correct_rows"] += int(pred == gold)
        stats["wrong_rows"] += int(bool(pred) and pred != gold)
        qtype = str(row.get("question_type") or row.get("subject") or "unknown")
        stats["by_question_type"].setdefault(qtype, {"total": 0, "wrong": 0})
        stats["by_question_type"][qtype]["total"] += 1
        stats["by_question_type"][qtype]["wrong"] += int(bool(pred) and pred != gold)
        if not pred or pred == gold:
            continue
        seen.add(key)
        is_multi = len(gold) > 1
        stats["multi_wrong_rows" if is_multi else "single_wrong_rows"] += 1
        rows.append(
            {
                "prompt": strip_instruction(prompt),
                "chosen": f"{ANSWER_PREFIX}{gold}",
                "rejected": f"{ANSWER_PREFIX}{pred}",
                "source": f"hard-exact-{row.get('source', 'unknown')}",
                "subject": qtype,
                "gold_answer": gold,
                "rejected_answer": pred,
                "is_multi": is_multi,
            }
        )
        if len(rows) >= args.max_rows:
            break

    written = write_jsonl(Path(args.out), rows)
    stats.update(
        {
            "out": args.out,
            "rows_written": written,
            "meets_min_rows": written >= args.min_rows,
            "single_rows_written": sum(1 for row in rows if not row["is_multi"]),
            "multi_rows_written": sum(1 for row in rows if row["is_multi"]),
            "multi_ratio_written": round(sum(1 for row in rows if row["is_multi"]) / written, 6) if written else 0.0,
        }
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if written < args.min_rows:
        raise SystemExit(f"Only built {written} preference rows, below --min-rows={args.min_rows}.")


if __name__ == "__main__":
    main()
