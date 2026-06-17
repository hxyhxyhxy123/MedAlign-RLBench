from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


GENERIC_PATTERNS = [
    "be careful",
    "slow down",
    "keep safe distance",
    "\u6ce8\u610f\u4f11\u606f",
    "\u53ca\u65f6\u5c31\u533b",
    "\u9075\u533b\u5631",
    "\u4fdd\u6301\u5b89\u5168",
]


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
        r"(?:\u6b63\u786e\u7b54\u6848|\u7b54\u6848|\u6700\u7ec8\u9009\u9879|\u9009\u62e9|ANSWER|CHOICE)\s*[:\uff1a\u4e3a\u662f-]*\s*([A-F](?:\s*[\u3001,， ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[\u3001,， ]?\s*[A-F]){0,5})\s*(?:$|\n)",
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


def available_options(prompt: str) -> set[str]:
    return set(re.findall(r"(?:^|\n)\s*([A-F])[\.\u3001\uff0e]", str(prompt).upper()))


def choice_f1(pred: str, gold: str) -> float:
    pred_set = set(pred)
    gold_set = set(gold)
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def score_sample(text: str, gold: str, prompt: str) -> dict[str, Any]:
    pred = extract_choice(text)
    exact = 1.0 if pred == gold else 0.0
    type_match = 1.0 if pred and ((len(pred) > 1) == (len(gold) > 1)) else 0.0
    has_format = bool(re.search(r"(\u7b54\u6848|ANSWER|CHOICE)\s*[:\uff1a]", text, flags=re.I))
    starts_choice = bool(re.match(r"^\s*[A-F](?:\s*[\u3001,， ]?\s*[A-F]){0,5}\b", text.upper()))
    format_reward = 1.0 if has_format else (0.6 if starts_choice else 0.0)
    opts = available_options(prompt)
    option_consistency = 1.0 if pred and (not opts or set(pred) <= opts) else 0.0
    too_short = 1.0 if compact_len(text) <= max(2, len(pred)) else 0.0
    generic = 1.0 if any(p.lower() in text.lower() for p in GENERIC_PATTERNS) else 0.0
    length_ok = 1.0 if 4 <= compact_len(text) <= 220 else 0.0
    reward = (
        0.45 * choice_f1(pred, gold)
        + 0.20 * exact
        + 0.15 * type_match
        + 0.10 * format_reward
        + 0.10 * option_consistency
        + 0.05 * length_ok
        - 0.05 * too_short
        - 0.05 * generic
    )
    return {
        "choice": pred,
        "reward": round(reward, 6),
        "choice_f1": round(choice_f1(pred, gold), 6),
        "exact": exact,
        "answer_type_match": type_match,
        "format_reward": format_reward,
        "option_consistency_reward": option_consistency,
        "too_short_penalty": too_short,
        "generic_answer_penalty": generic,
        "length_ok": length_ok,
    }


def ensure_response(text: str, choice: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return f"\u7b54\u6848\uff1a{choice}"
    if re.search(r"(\u7b54\u6848|ANSWER|CHOICE)\s*[:\uff1a]", cleaned, flags=re.I):
        return cleaned
    return f"\u7b54\u6848\uff1a{choice}\n\u89e3\u6790\uff1a{cleaned}"


def oracle_response(gold: str) -> str:
    return (
        f"\u7b54\u6848\uff1a{gold}\n"
        f"\u89e3\u6790\uff1a\u6839\u636e\u9898\u5e72\u5173\u952e\u6761\u4ef6\u548c\u9009\u9879\u5bf9\u6bd4\uff0c"
        f"\u6b63\u786e\u9009\u9879\u4e3a {gold}\u3002"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--min-rows", type=int, default=1500)
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--allow-oracle-chosen", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.samples))
    pref_rows: list[dict[str, Any]] = []
    scored_prompts = 0
    prompts_with_sampled_pair = 0
    prompts_with_oracle_pair = 0
    skipped_no_gold = 0
    skipped_no_negative = 0
    skipped_margin = 0
    total_samples = 0
    correct_samples = 0
    wrong_samples = 0

    for row in rows:
        gold = normalize_choice(row.get("gold_choice") or row.get("answer", ""))
        if not gold:
            skipped_no_gold += 1
            continue
        prompt = str(row.get("prompt") or row.get("input") or "")
        samples = list(row.get("samples") or [])
        scored: list[dict[str, Any]] = []
        for sample in samples:
            text = str(sample.get("text", ""))
            score = score_sample(text, gold, prompt)
            merged = dict(sample)
            merged.update(score)
            scored.append(merged)
            total_samples += 1
            correct_samples += int(score["choice"] == gold)
            wrong_samples += int(score["choice"] != gold)
        if not scored:
            continue
        scored_prompts += 1

        positives = [s for s in scored if s["choice"] == gold]
        negatives = [s for s in scored if s["choice"] != gold]
        if not negatives:
            skipped_no_negative += 1
            continue

        used_oracle = False
        if positives:
            chosen_sample = max(positives, key=lambda s: (s["reward"], len(str(s.get("text", "")))))
            chosen_text = ensure_response(str(chosen_sample.get("text", "")), gold)
            chosen_reward = float(chosen_sample["reward"])
        elif args.allow_oracle_chosen:
            chosen_sample = {"reward": 1.0, "choice": gold, "text": oracle_response(gold)}
            chosen_text = oracle_response(gold)
            chosen_reward = 1.0
            used_oracle = True
        else:
            continue

        rejected_sample = min(negatives, key=lambda s: (s["reward"], -len(str(s.get("text", "")))))
        margin = chosen_reward - float(rejected_sample["reward"])
        if margin < args.min_margin:
            skipped_margin += 1
            continue

        rejected_choice = str(rejected_sample.get("choice", ""))
        rejected_text = ensure_response(str(rejected_sample.get("text", "")), rejected_choice or "?")
        pref_rows.append(
            {
                "prompt": f"\u8bf7\u56de\u7b54\u4e0b\u9762\u7684\u4e2d\u6587\u533b\u5b66\u8003\u8bd5\u9898\uff0c\u5e76\u7ed9\u51fa\u7b80\u77ed\u89e3\u6790\u3002\n{prompt}",
                "chosen": chosen_text,
                "rejected": rejected_text,
                "source": "rs-hard-cmexam",
                "subject": str(row.get("exam_subject") or row.get("subject") or row.get("question_type") or "unknown"),
                "gold_answer": gold,
                "greedy_answer": str(row.get("greedy_choice", "")),
                "rejected_answer": rejected_choice,
                "chosen_reward": round(chosen_reward, 6),
                "rejected_reward": rejected_sample["reward"],
                "reward_margin": round(margin, 6),
                "oracle_chosen": used_oracle,
            }
        )
        prompts_with_sampled_pair += int(not used_oracle)
        prompts_with_oracle_pair += int(used_oracle)
        if len(pref_rows) >= args.max_rows:
            break

    write_jsonl(Path(args.out), pref_rows)
    summary = {
        "samples": args.samples,
        "out": args.out,
        "sample_rows": len(rows),
        "scored_prompts": scored_prompts,
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "wrong_samples": wrong_samples,
        "sample_accuracy": correct_samples / total_samples if total_samples else 0.0,
        "preference_rows": len(pref_rows),
        "sampled_chosen_rows": prompts_with_sampled_pair,
        "oracle_chosen_rows": prompts_with_oracle_pair,
        "oracle_chosen_rate": prompts_with_oracle_pair / len(pref_rows) if pref_rows else 0.0,
        "min_rows_requested": args.min_rows,
        "meets_min_rows": len(pref_rows) >= args.min_rows,
        "min_margin": args.min_margin,
        "skipped_no_gold": skipped_no_gold,
        "skipped_no_negative": skipped_no_negative,
        "skipped_margin": skipped_margin,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(pref_rows) < args.min_rows:
        raise SystemExit(f"Only built {len(pref_rows)} RS preference rows, below --min-rows={args.min_rows}.")


if __name__ == "__main__":
    main()
