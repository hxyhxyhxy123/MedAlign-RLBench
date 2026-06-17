from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value).upper()))))


def extract_choice(text: Any) -> str:
    raw = str(text or "").strip()
    upper = raw.upper()
    patterns = [
        r"(?:正确答案|答案|答|选项|选择|ANSWER|CHOICE)\s*[:：为是\-]*\s*([A-F](?:\s*[、,，/ ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[、,，/ ]?\s*[A-F]){0,5})\s*(?:$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            value = normalize_choice(match.group(1))
            if value:
                return value
    compact = re.sub(r"\s+", "", upper)
    match = re.search(r"(?<![A-Z])([A-F]{1,6})(?![A-Z])", compact)
    return normalize_choice(match.group(1)) if match else ""


def format_response(answer: str, reason: str) -> str:
    return f"答案：{answer}\n解析：{reason}".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--min-rows", type=int, default=3000)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    hard_rows: list[dict[str, Any]] = []
    total = 0
    correct = 0
    wrong = 0
    invalid = 0
    multi_total = 0
    multi_wrong = 0
    by_source: dict[str, dict[str, int]] = {}
    by_qtype: dict[str, dict[str, int]] = {}

    for row in rows:
        gold = normalize_choice(row.get("answer", ""))
        if not gold:
            continue
        pred_text = str(row.get("prediction", ""))
        pred = extract_choice(pred_text)
        total += 1
        ok = pred == gold
        correct += int(ok)
        wrong += int(not ok)
        invalid += int(not pred)
        if len(gold) > 1:
            multi_total += 1
            multi_wrong += int(not ok)
        source = str(row.get("source", "unknown"))
        qtype = str(row.get("question_type", "unknown"))
        by_source.setdefault(source, {"total": 0, "wrong": 0})
        by_qtype.setdefault(qtype, {"total": 0, "wrong": 0})
        by_source[source]["total"] += 1
        by_source[source]["wrong"] += int(not ok)
        by_qtype[qtype]["total"] += 1
        by_qtype[qtype]["wrong"] += int(not ok)
        if ok or (not pred and not args.include_invalid):
            continue

        prompt = str(row.get("prompt") or row.get("input") or "")
        if not prompt:
            continue
        chosen = format_response(gold, f"该题正确答案为 {gold}，需要结合题干条件、选项差异和医学知识判断。")
        if pred:
            rejected = format_response(pred, f"这是模型在候选题上的错误回答；它与标准答案 {gold} 不一致，应作为 hard negative。")
        else:
            rejected = pred_text.strip() or "无法给出有效选项。"
        hard_rows.append(
            {
                "prompt": f"请回答下面的中文医学考试题，并给出简短解析。\n{prompt}",
                "chosen": chosen,
                "rejected": rejected,
                "source": f"hard-{source}",
                "subject": str(row.get("exam_subject") or row.get("subject") or qtype),
                "gold_answer": gold,
                "model_answer": pred,
                "question_type": qtype,
            }
        )
        if len(hard_rows) >= args.max_rows:
            break

    write_jsonl(Path(args.out), hard_rows)
    summary = {
        "predictions": args.predictions,
        "out": args.out,
        "prediction_rows": len(rows),
        "scored_rows": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "invalid_prediction_count": invalid,
        "invalid_prediction_rate": invalid / total if total else 0.0,
        "multi_choice_total": multi_total,
        "multi_choice_wrong": multi_wrong,
        "hard_preference_rows": len(hard_rows),
        "min_rows_requested": args.min_rows,
        "meets_min_rows": len(hard_rows) >= args.min_rows,
        "by_source": by_source,
        "by_question_type": by_qtype,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(hard_rows) < args.min_rows:
        raise SystemExit(f"Only built {len(hard_rows)} hard preference rows, below --min-rows={args.min_rows}.")


if __name__ == "__main__":
    main()
