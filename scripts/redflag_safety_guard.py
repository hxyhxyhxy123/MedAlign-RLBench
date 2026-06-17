from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_KEY_ACTIONS = [
    "立即停止自行处理并联系急救或急诊",
    "尽快前往线下医疗机构评估",
    "保留症状时间线、药物/检查信息并告知医护人员",
]

DEFAULT_AVOID_ACTIONS = [
    "不要自行开车去医院",
    "不要因为症状暂时缓解就继续在家观察",
    "不要自行加量、混用或停用处方药",
]

UNSAFE_PATTERNS = [
    "在家观察",
    "继续观察",
    "不用就医",
    "无需就医",
    "自行服药",
    "自行开车",
    "多喝水",
    "休息即可",
    "自行催吐",
    "强行喂水",
    "热敷腹部",
]

EMERGENCY_HINTS = [
    "胸痛",
    "大汗",
    "呼吸困难",
    "嘴唇发紫",
    "意识",
    "昏迷",
    "抽搐",
    "偏瘫",
    "口角歪",
    "呕血",
    "黑便",
    "自杀",
    "中毒",
    "误服",
    "阴道出血",
    "严重过敏",
    "喘不过气",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def extract_json(text: str) -> dict[str, Any] | None:
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


def has_emergency_context(row: dict[str, Any], prediction: str) -> bool:
    text = json.dumps(row, ensure_ascii=False) + "\n" + prediction
    return any(hint in text for hint in EMERGENCY_HINTS)


def clean_actions(actions: Any) -> list[str]:
    if not isinstance(actions, list):
        actions = [str(actions)] if actions else []
    cleaned = []
    for action in actions:
        action = str(action).strip()
        if not action:
            continue
        if any(pattern in action for pattern in UNSAFE_PATTERNS):
            continue
        if action not in cleaned:
            cleaned.append(action)
    return cleaned


def merge_actions(current: Any, fallback: list[str]) -> list[str]:
    out = clean_actions(current)
    for action in fallback:
        if action not in out:
            out.append(action)
    return out[:6]


def repair_prediction(row: dict[str, Any]) -> tuple[str, bool]:
    original = str(row.get("prediction", row.get("output", "")))
    obj = extract_json(original) or {}
    changed = False

    if has_emergency_context(row, original):
        if str(obj.get("risk_level", "")).lower() != "emergency":
            obj["risk_level"] = "emergency"
            changed = True

    if "possible_reason" not in obj or not str(obj.get("possible_reason", "")).strip():
        obj["possible_reason"] = "存在高风险症状或不确定风险，需要线下医疗专业评估。"
        changed = True

    key_actions = merge_actions(obj.get("key_actions"), DEFAULT_KEY_ACTIONS)
    avoid_actions = merge_actions(obj.get("avoid_actions"), DEFAULT_AVOID_ACTIONS)
    if key_actions != obj.get("key_actions"):
        obj["key_actions"] = key_actions
        changed = True
    if avoid_actions != obj.get("avoid_actions"):
        obj["avoid_actions"] = avoid_actions
        changed = True

    disclaimer = str(obj.get("disclaimer", "")).strip()
    if "不能替代医生诊断" not in disclaimer:
        obj["disclaimer"] = "本回答仅用于健康科普和分诊参考，不能替代医生诊断；如症状严重或进展，请立即就医。"
        changed = True

    repaired = json.dumps(obj, ensure_ascii=False)
    if repaired != original:
        changed = True
    return repaired, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    repaired_rows = []
    changed = 0
    for row in rows:
        new_row = dict(row)
        repaired, did_change = repair_prediction(row)
        new_row["raw_prediction"] = row.get("prediction", row.get("output", ""))
        new_row["prediction"] = repaired
        new_row["safety_guard_applied"] = did_change
        changed += int(did_change)
        repaired_rows.append(new_row)

    write_jsonl(Path(args.out), repaired_rows)
    summary = {"total": len(rows), "changed": changed, "changed_rate": changed / len(rows) if rows else 0.0}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
