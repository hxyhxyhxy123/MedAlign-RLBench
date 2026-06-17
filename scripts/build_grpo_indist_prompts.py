"""Build IN-DISTRIBUTION GRPO prompts from CMB-test by carving a disjoint split.

The full CMB-test (11200) is split by exact question text into:
  - held-out eval  = the existing 3000 random-noleak screen set (NEVER trained on)
  - GRPO train     = the remaining ~8200 rows (used as RL prompts)

This removes the CMExam->CMB distribution shift that made every prior preference
run underperform SFT. Strict text-key dedup guarantees zero train/eval overlap.
Prompt format matches scripts/generate_eval_predictions_sharded.py exactly.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

SYSTEM = "你是严谨的中文医学考试助手。请只输出最终答案选项，例如 A 或 BCDEF，不要输出解析。"


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value or "").upper()))))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", "", str(prompt or ""))


def load_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in iter_jsonl(path):
        p = str(row.get("prompt") or row.get("input") or row.get("question") or "")
        if p:
            keys.add(prompt_key(p))
    return keys


def make_row(prompt: str, gold: str, is_multi: bool, source: str) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": str(prompt).strip()},
        ],
        "gold_answer": gold,
        "is_multi": is_multi,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True, help="full CMB-test choice eval jsonl (11200)")
    parser.add_argument("--exclude", nargs="*", default=[], help="held-out eval files to exclude by text key")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--multi-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exclude_keys: set[str] = set()
    for p in args.exclude:
        exclude_keys |= load_keys(Path(p))

    rng = random.Random(args.seed)
    multi_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"full_rows": 0, "excluded_overlap": 0, "dup": 0, "no_gold": 0, "kept": 0}

    for row in iter_jsonl(Path(args.full)):
        stats["full_rows"] += 1
        prompt = str(row.get("prompt") or row.get("input") or row.get("question") or "")
        gold = normalize_choice(row.get("answer") or row.get("gold_answer") or "")
        if not prompt or not gold:
            stats["no_gold"] += 1
            continue
        key = prompt_key(prompt)
        if key in exclude_keys:
            stats["excluded_overlap"] += 1
            continue
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        is_multi = len(gold) > 1
        built = make_row(prompt, gold, is_multi, str(row.get("source") or "CMB-train-split"))
        (multi_rows if is_multi else single_rows).append(built)
        stats["kept"] += 1

    rows: list[dict[str, Any]] = []
    for r in multi_rows:
        for _ in range(max(1, args.multi_repeat)):
            rows.append(dict(r))
    rng.shuffle(single_rows)
    remaining = max(0, args.max_rows - len(rows))
    rows.extend(single_rows[:remaining])
    rng.shuffle(rows)
    rows = rows[: args.max_rows]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "full": args.full,
        "exclude": args.exclude,
        "exclude_key_count": len(exclude_keys),
        **stats,
        "out": args.out,
        "rows_written": len(rows),
        "multi_source_rows": len(multi_rows),
        "single_source_rows": len(single_rows),
        "multi_rows_written": sum(1 for r in rows if r["is_multi"]),
        "single_rows_written": sum(1 for r in rows if not r["is_multi"]),
        "multi_ratio_written": round(sum(1 for r in rows if r["is_multi"]) / len(rows), 6) if rows else 0.0,
        "multi_repeat": args.multi_repeat,
        "max_rows": args.max_rows,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(rows) < 1000:
        raise SystemExit(f"Only built {len(rows)} in-distribution prompts; expected >1000.")


if __name__ == "__main__":
    main()
