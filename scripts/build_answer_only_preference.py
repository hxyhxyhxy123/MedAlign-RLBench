from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


INSTRUCTION_PREFIXES = [
    "请回答下面的中文医学考试题，并给出简短解析。",
    "请回答下面的中文医学考试题，并给出简短解析。\n",
    "请只输出最终答案选项，例如 A 或 BCDEF，不要输出解析。",
    "请只输出最终答案选项（例如 A 或 BCDEF），不要输出解析。",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value).upper()))))


def extract_choice(text: Any) -> str:
    raw = str(text or "").strip()
    upper = raw.upper()
    patterns = [
        r"(?:正确答案|答案|最终选项|选择|ANSWER|CHOICE)\s*[:：为是-]*\s*([A-F](?:\s*[、,， ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[、,， ]?\s*[A-F]){0,5})\s*(?:$|\n)",
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


def strip_instruction(prompt: str) -> str:
    text = str(prompt or "").strip()
    for prefix in INSTRUCTION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def prompt_key(prompt: str) -> str:
    text = strip_instruction(prompt)
    text = re.sub(r"\s+", "", text)
    return text


def available_options(prompt: str) -> list[str]:
    opts = sorted(set(re.findall(r"(?:^|\n)\s*([A-F])[\.\u3001\uff0e]", str(prompt).upper())))
    return opts or list("ABCDE")


def parse_training_row(row: dict[str, Any]) -> tuple[str, str, str, str]:
    if "input" in row or "output" in row:
        question = str(row.get("input") or row.get("prompt") or "")
        answer = extract_choice(row.get("output") or row.get("answer") or "")
        source = str(row.get("source") or "sft")
        qtype = str(row.get("question_type") or row.get("subject") or "")
        return strip_instruction(question), answer, source, qtype

    question = str(row.get("prompt") or row.get("question") or "")
    answer = normalize_choice(row.get("answer") or row.get("gold_answer") or "")
    if not answer:
        answer = extract_choice(row.get("chosen") or row.get("output") or "")
    source = str(row.get("source") or "unknown")
    qtype = str(row.get("question_type") or row.get("subject") or "")
    return strip_instruction(question), answer, source, qtype


def load_exclude_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            prompt = str(row.get("prompt") or row.get("input") or "")
            if prompt:
                keys.add(prompt_key(prompt))
    return keys


def load_prediction_negatives(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    negatives: dict[str, str] = {}
    for row in iter_jsonl(path):
        question = str(row.get("prompt") or row.get("input") or "")
        gold = normalize_choice(row.get("answer") or row.get("gold_choice") or "")
        pred = extract_choice(row.get("prediction") or row.get("model_answer") or "")
        if question and gold and pred and pred != gold:
            negatives[prompt_key(question)] = pred
    return negatives


def mutate_multi(gold: str, opts: list[str], rng: random.Random) -> list[str]:
    gold_set = set(gold)
    wrongs: list[str] = []
    if len(gold) > 1:
        for item in sorted(gold_set):
            candidate = "".join(sorted(gold_set - {item}))
            if candidate:
                wrongs.append(candidate)
    for opt in opts:
        if opt not in gold_set:
            candidate = "".join(sorted(gold_set | {opt}))
            if candidate != gold:
                wrongs.append(candidate)
    if not wrongs:
        wrongs = [opt for opt in opts if opt != gold]
    rng.shuffle(wrongs)
    return list(dict.fromkeys(wrongs))


def build_negatives(gold: str, opts: list[str], preferred: str, pairs_per_question: int, rng: random.Random) -> list[str]:
    candidates: list[str] = []
    if preferred and preferred != gold and set(preferred) <= set(opts):
        candidates.append(preferred)
    if len(gold) > 1:
        candidates.extend(mutate_multi(gold, opts, rng))
    else:
        wrong_opts = [opt for opt in opts if opt != gold]
        rng.shuffle(wrong_opts)
        candidates.extend(wrong_opts)
    filtered: list[str] = []
    for candidate in candidates:
        norm = normalize_choice(candidate)
        if norm and norm != gold and set(norm) <= set(opts) and norm not in filtered:
            filtered.append(norm)
    return filtered[:pairs_per_question]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--exclude-eval", nargs="*", default=[])
    parser.add_argument("--predictions", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument("--pairs-per-question", type=int, default=2)
    parser.add_argument("--multi-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    exclude_keys = load_exclude_keys([Path(p) for p in args.exclude_eval])
    prediction_negatives = load_prediction_negatives(Path(args.predictions) if args.predictions else None)

    seen_questions: set[str] = set()
    candidates: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "source_files": args.sources,
        "exclude_eval": args.exclude_eval,
        "exclude_key_count": len(exclude_keys),
        "prediction_negative_count": len(prediction_negatives),
        "read_rows": 0,
        "usable_questions": 0,
        "skipped_overlap": 0,
        "skipped_duplicate": 0,
        "skipped_no_answer": 0,
        "skipped_no_options": 0,
        "multi_questions": 0,
        "single_questions": 0,
        "rows_from_prediction_negative": 0,
        "rows_by_source": {},
    }

    for source_file in args.sources:
        path = Path(source_file)
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            stats["read_rows"] += 1
            question, gold, source, qtype = parse_training_row(row)
            gold = normalize_choice(gold)
            if not question or not gold:
                stats["skipped_no_answer"] += 1
                continue
            key = prompt_key(question)
            if key in exclude_keys:
                stats["skipped_overlap"] += 1
                continue
            if key in seen_questions:
                stats["skipped_duplicate"] += 1
                continue
            opts = available_options(question)
            if not opts or not set(gold) <= set(opts):
                stats["skipped_no_options"] += 1
                continue
            seen_questions.add(key)
            stats["usable_questions"] += 1
            is_multi = len(gold) > 1
            stats["multi_questions" if is_multi else "single_questions"] += 1
            preferred_negative = prediction_negatives.get(key, "")
            negatives = build_negatives(gold, opts, preferred_negative, args.pairs_per_question, rng)
            repeat = args.multi_repeat if is_multi else 1
            for neg in negatives:
                for _ in range(repeat):
                    candidates.append(
                        {
                            "prompt": question,
                            "chosen": f"答案：{gold}",
                            "rejected": f"答案：{neg}",
                            "source": f"answer-only-{source}",
                            "subject": qtype,
                            "gold_answer": gold,
                            "rejected_answer": neg,
                            "is_multi": is_multi,
                            "from_prediction_negative": bool(preferred_negative and neg == preferred_negative),
                        }
                    )
                    stats["rows_from_prediction_negative"] += int(bool(preferred_negative and neg == preferred_negative))
                    stats["rows_by_source"].setdefault(source, 0)
                    stats["rows_by_source"][source] += 1

    hard = [row for row in candidates if row["from_prediction_negative"]]
    multi = [row for row in candidates if row["is_multi"] and not row["from_prediction_negative"]]
    other = [row for row in candidates if not row["is_multi"] and not row["from_prediction_negative"]]
    rng.shuffle(hard)
    rng.shuffle(multi)
    rng.shuffle(other)
    ordered = hard + multi + other
    selected = ordered[: args.max_rows]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.update(
        {
            "candidate_preference_rows": len(candidates),
            "selected_preference_rows": len(selected),
            "selected_multi_rows": sum(1 for row in selected if row["is_multi"]),
            "selected_prediction_negative_rows": sum(1 for row in selected if row["from_prediction_negative"]),
            "max_rows": args.max_rows,
            "pairs_per_question": args.pairs_per_question,
            "multi_repeat": args.multi_repeat,
        }
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
