from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any


def safe_text(value: Any, limit: int = 4000) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def format_prompt(row: dict[str, Any]) -> str:
    opts = row.get("option") if isinstance(row.get("option"), dict) else {}
    option_lines = [f"{k}. {safe_text(opts[k], 500)}" for k in sorted(opts)]
    meta = " / ".join(
        safe_text(row.get(k), 80)
        for k in ["exam_type", "exam_class", "exam_subject", "question_type"]
        if row.get(k)
    )
    return "\n".join(x for x in [f"题目来源：{meta}" if meta else "", safe_text(row.get("question"), 1600), "\n".join(option_lines)] if x)


def main() -> None:
    out = Path("data/eval")
    zip_path = Path("baselines/CMB/data/CMB.zip")
    answer_path = Path("baselines/CMB/data/CMB-test-choice-answer.json")
    if not zip_path.exists() or not answer_path.exists():
        raise FileNotFoundError("CMB baseline data is missing.")

    answers = {int(r["id"]): r for r in json.loads(answer_path.read_text(encoding="utf-8"))}
    with zipfile.ZipFile(zip_path) as zf:
        test_questions = json.loads(zf.read("CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json").decode("utf-8"))
        val_questions = json.loads(zf.read("CMB/CMB-Exam/CMB-val/CMB-val-merge.json").decode("utf-8"))
        clin_cases = json.loads(zf.read("CMB/CMB-Clin/CMB-Clin-qa.json").decode("utf-8"))

    test_rows = []
    for row in test_questions:
        rid = int(row["id"])
        ans = answers.get(rid, {})
        merged = dict(row)
        merged["answer"] = ans.get("answer", "")
        test_rows.append(
            {
                "id": rid,
                "prompt": format_prompt(merged),
                "answer": merged["answer"],
                "exam_type": merged.get("exam_type"),
                "exam_class": merged.get("exam_class"),
                "exam_subject": merged.get("exam_subject"),
                "question_type": merged.get("question_type"),
                "source": "CMB-test",
            }
        )

    val_rows = []
    for idx, row in enumerate(val_questions, start=1):
        val_rows.append(
            {
                "id": idx,
                "prompt": format_prompt(row),
                "answer": row.get("answer", ""),
                "explanation": safe_text(row.get("explanation"), 2000),
                "exam_type": row.get("exam_type"),
                "exam_class": row.get("exam_class"),
                "exam_subject": row.get("exam_subject"),
                "question_type": row.get("question_type"),
                "source": "CMB-val",
            }
        )

    clin_rows = []
    for case in clin_cases:
        desc = safe_text(case.get("description"), 3500)
        title = safe_text(case.get("title"), 200)
        for i, qa in enumerate(case.get("QA_pairs", []), start=1):
            clin_rows.append(
                {
                    "id": f"{case.get('id')}-{i}",
                    "prompt": f"{title}\n{desc}\n问题：{safe_text(qa.get('question'), 600)}",
                    "answer": safe_text(qa.get("answer"), 2400),
                    "source": "CMB-Clin",
                    "case_title": title,
                }
            )

    redflag_rows = []
    redflag_path = Path("data/stage1/stage1_redflag_dpo.jsonl")
    if redflag_path.exists():
        with redflag_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                row = json.loads(line)
                redflag_rows.append({"id": idx, **row, "source": "manual-redflag"})

    write_jsonl(out / "cmb_test_choice_eval.jsonl", test_rows)
    write_jsonl(out / "cmb_val_choice_eval.jsonl", val_rows)
    write_jsonl(out / "cmb_clin_eval.jsonl", clin_rows)
    write_jsonl(out / "redflag_eval.jsonl", redflag_rows)
    summary = {
        "cmb_test_choice_eval": len(test_rows),
        "cmb_val_choice_eval": len(val_rows),
        "cmb_clin_eval": len(clin_rows),
        "redflag_eval": len(redflag_rows),
        "cmb_test_missing_answers": sum(1 for r in test_rows if not r["answer"]),
    }
    write_json(out / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
