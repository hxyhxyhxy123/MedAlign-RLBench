from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hard_exact_preference import strip_instruction

SYSTEM = "你是严谨的中文医学考试助手。请只输出最终答案选项，例如 A 或 BCDEF，不要输出解析。"
CHOICE_INSTRUCTION = "请只输出最终答案选项（例如 A 或 BCDEF），不要输出解析。"


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value or "").upper()))))


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


def make_row(prompt: str, gold: str, is_multi: bool, source: str, sft_correct: bool | None) -> dict[str, Any]:
    # Match the eval prompt EXACTLY: system carries the instruction, user is the
    # raw question only. Any extra instruction line here would create a train/eval
    # mismatch that drifts the policy.
    user = strip_instruction(prompt)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "gold_answer": gold,
        "is_multi": is_multi,
        "sft_correct": sft_correct,
        "source": source,
    }


def extract_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (multi_rows, single_rows) from a candidate-prediction or preference file."""
    multi_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        prompt = str(row.get("prompt") or "").strip()
        gold = normalize_choice(row.get("answer") or row.get("gold_answer") or row.get("chosen") or "")
        if not prompt or not gold:
            continue
        key = re.sub(r"\s+", "", prompt)
        if key in seen:
            continue
        seen.add(key)
        is_multi = bool(row.get("is_multi")) or len(gold) > 1
        sft_correct: bool | None = None
        if "prediction" in row:
            pred = normalize_choice(re.sub(r"^答案[:：]?", "", str(row.get("prediction") or "")))
            sft_correct = bool(pred) and pred == gold
        built = make_row(prompt, gold, is_multi, str(row.get("source") or "candidate"), sft_correct)
        (multi_rows if is_multi else single_rows).append(built)
    return multi_rows, single_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="12000 candidate prediction jsonl (preferred source)")
    parser.add_argument("--preference", help="fallback preference jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--multi-repeat", type=int, default=3, help="duplicate multi-choice rows")
    parser.add_argument("--drop-easy-frac", type=float, default=0.0,
                        help="randomly drop this fraction of SFT-correct single-choice rows to focus on hard ones")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.predictions or args.preference)
    if not source.exists():
        raise SystemExit(f"source not found: {source}")

    rng = random.Random(args.seed)
    multi_rows, single_rows = extract_rows(source)

    if args.drop_easy_frac > 0:
        kept_single = []
        for r in single_rows:
            if r.get("sft_correct") and rng.random() < args.drop_easy_frac:
                continue
            kept_single.append(r)
        single_rows = kept_single

    rows: list[dict[str, Any]] = []
    for row in multi_rows:
        for _ in range(max(1, args.multi_repeat)):
            rows.append(dict(row))

    rng.shuffle(single_rows)
    remaining = max(0, args.max_rows - len(rows))
    rows.extend(single_rows[:remaining])
    rng.shuffle(rows)
    rows = rows[: args.max_rows]

    written = write_jsonl(Path(args.out), rows)
    summary = {
        "source": str(source),
        "out": args.out,
        "rows_written": written,
        "multi_source_rows": len(multi_rows),
        "single_source_rows": len(single_rows),
        "multi_rows_written": sum(1 for r in rows if r["is_multi"]),
        "single_rows_written": sum(1 for r in rows if not r["is_multi"]),
        "multi_ratio_written": round(sum(1 for r in rows if r["is_multi"]) / written, 6) if written else 0.0,
        "sft_correct_written": sum(1 for r in rows if r.get("sft_correct") is True),
        "sft_wrong_written": sum(1 for r in rows if r.get("sft_correct") is False),
        "multi_repeat": args.multi_repeat,
        "drop_easy_frac": args.drop_easy_frac,
        "max_rows": args.max_rows,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if written < 256:
        raise SystemExit(f"Only built {written} GRPO prompts; need at least 256.")


if __name__ == "__main__":
    main()
