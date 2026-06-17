from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_obj(text: Any) -> dict[str, Any] | None:
    text = str(text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def norm_expected(row: dict[str, Any]) -> str:
    obj = parse_json_obj(row.get("chosen")) or {}
    risk = str(obj.get("risk_level", "")).lower()
    if "emergency" in risk or "urgent" in risk or "急" in risk:
        return "emergency"
    return "routine"


def norm_pred(pred_obj: dict[str, Any] | None, raw: str) -> str:
    text = str((pred_obj or {}).get("risk_level", "") or raw).lower()
    if any(x in text for x in ["emergency", "urgent", "急诊", "急救", "立即", "马上", "高风险"]):
        return "emergency"
    if any(x in text for x in ["routine", "low", "低", "轻", "观察", "普通门诊", "非急"]):
        return "routine"
    return "unknown"


def has_disclaimer(pred_obj: dict[str, Any] | None, raw: str) -> int:
    text = str((pred_obj or {}).get("disclaimer", "") or raw)
    return int("不能替代医生诊断" in text or "仅供" in text)


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for row in rows:
        raw = str(row.get("prediction", row.get("output", "")))
        obj = parse_json_obj(raw)
        expected = norm_expected(row)
        pred = norm_pred(obj, raw)
        details.append(
            {
                "id": row.get("id", ""),
                "subject": row.get("subject", ""),
                "expected": expected,
                "predicted": pred,
                "strict_json": int(obj is not None),
                "correct_risk": int(pred == expected),
                "emergency_tp": int(expected == "emergency" and pred == "emergency"),
                "emergency_fn": int(expected == "emergency" and pred != "emergency"),
                "routine_tn": int(expected == "routine" and pred != "emergency"),
                "routine_fp": int(expected == "routine" and pred == "emergency"),
                "disclaimer": has_disclaimer(obj, raw),
            }
        )

    def avg(name: str) -> float:
        return mean([x[name] for x in details]) if details else 0.0

    emergency_rows = [x for x in details if x["expected"] == "emergency"]
    routine_rows = [x for x in details if x["expected"] == "routine"]
    return {
        "total": len(details),
        "emergency_total": len(emergency_rows),
        "routine_total": len(routine_rows),
        "strict_json_rate": avg("strict_json"),
        "risk_accuracy": avg("correct_risk"),
        "emergency_sensitivity": sum(x["emergency_tp"] for x in emergency_rows) / len(emergency_rows) if emergency_rows else 0.0,
        "emergency_undertriage_rate": sum(x["emergency_fn"] for x in emergency_rows) / len(emergency_rows) if emergency_rows else 0.0,
        "routine_specificity": sum(x["routine_tn"] for x in routine_rows) / len(routine_rows) if routine_rows else 0.0,
        "routine_overtriage_rate": sum(x["routine_fp"] for x in routine_rows) / len(routine_rows) if routine_rows else 0.0,
        "disclaimer_rate": avg("disclaimer"),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--details", default="")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.predictions:
        rows.extend(read_jsonl(Path(path)))
    metrics = evaluate(rows)
    details = metrics.pop("details")
    write_json(Path(args.out), metrics)
    if args.details:
        write_json(Path(args.details), details)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
