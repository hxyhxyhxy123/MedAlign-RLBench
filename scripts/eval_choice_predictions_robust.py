from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_choice(value: str) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value).upper()))))


def extract_choice(text: str) -> str:
    raw = str(text).strip()
    upper = raw.upper()
    label_patterns = [
        r"(?:正确答案|答案|答|选项|选择|ANSWER|CHOICE)\s*[:：为是\-]*\s*([A-F](?:\s*[、,，/ ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[、,，/ ]?\s*[A-F]){0,5})\s*(?:$|\n)",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, upper)
        if match:
            value = normalize_choice(match.group(1))
            if value:
                return value

    compact = re.sub(r"\s+", "", upper)
    isolated = re.search(r"(?<![A-Z])([A-F]{1,6})(?![A-Z])", compact)
    if isolated:
        return normalize_choice(isolated.group(1))
    return ""


def eval_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    invalid = 0
    multi_total = 0
    multi_correct = 0
    single_total = 0
    single_correct = 0
    by_source: dict[str, list[int]] = defaultdict(list)
    by_question_type: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        gold = normalize_choice(str(row.get("answer", "")))
        if not gold:
            continue
        pred = extract_choice(str(row.get("prediction", row.get("output", ""))))
        ok = int(pred == gold)
        total += 1
        correct += ok
        invalid += int(not pred)

        if len(gold) > 1:
            multi_total += 1
            multi_correct += ok
        else:
            single_total += 1
            single_correct += ok

        by_source[str(row.get("source", "unknown"))].append(ok)
        by_question_type[str(row.get("question_type", "unknown"))].append(ok)

    def summarize(values: list[int]) -> dict[str, Any]:
        return {
            "total": len(values),
            "correct": int(sum(values)),
            "accuracy": float(sum(values) / len(values)) if values else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid_prediction_count": invalid,
        "invalid_prediction_rate": invalid / total if total else 0.0,
        "single_choice": {
            "total": single_total,
            "correct": single_correct,
            "accuracy": single_correct / single_total if single_total else 0.0,
        },
        "multi_choice": {
            "total": multi_total,
            "correct": multi_correct,
            "accuracy": multi_correct / multi_total if multi_total else 0.0,
        },
        "by_source": {key: summarize(values) for key, values in sorted(by_source.items())},
        "by_question_type": {key: summarize(values) for key, values in sorted(by_question_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metrics = eval_choice(read_jsonl(Path(args.predictions)))
    write_json(Path(args.out), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
