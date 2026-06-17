from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False)
    return ""


def ceval_to_examples(row: dict[str, Any]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    q = first_text(row, ["question", "query", "stem"])
    if not q:
        return None, None
    options = []
    for k in ["A", "B", "C", "D", "E"]:
        if row.get(k):
            options.append(f"{k}. {row[k]}")
    answer = str(row.get("answer", row.get("label", ""))).strip()
    if not answer:
        return None, None
    input_text = q + "\n" + "\n".join(options)
    chosen = f"答案：{answer}\n理由：根据题干和医学知识，最符合要求的选项是 {answer}。"
    wrong = "A" if answer != "A" else "B"
    rejected = f"答案：{wrong}\n理由：该选项更合适。"
    sft = {"instruction": "请回答下面的医学选择题，并给出简短理由。", "input": input_text, "output": chosen}
    dpo = {"prompt": f"请回答下面的医学选择题，并给出简短理由。\n{input_text}", "chosen": chosen, "rejected": rejected}
    return sft, dpo


def cmb_to_examples(row: dict[str, Any]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    question = first_text(row, ["question", "query", "stem"])
    if not question:
        return None, None
    option = row.get("option") or {}
    options = []
    if isinstance(option, dict):
        for key in sorted(option):
            options.append(f"{key}. {option[key]}")
    answer = str(row.get("answer", "")).strip()
    explanation = str(row.get("explanation", "")).strip()
    if not answer:
        return None, None
    meta = " / ".join(str(row.get(k, "")).strip() for k in ["exam_type", "exam_class", "exam_subject"] if row.get(k))
    prompt_body = question + ("\n" + "\n".join(options) if options else "")
    if meta:
        prompt_body = f"题目来源：{meta}\n{prompt_body}"
    reason = explanation if explanation else f"根据题干与医学知识，正确答案为 {answer}。"
    chosen = f"答案：{answer}\n解析：{reason}"
    wrong_candidates = [k for k in ["A", "B", "C", "D", "E"] if k not in set(answer)]
    wrong = wrong_candidates[0] if wrong_candidates else "A"
    rejected = f"答案：{wrong}\n解析：该回答未能正确匹配题干中的医学知识点，存在误导风险。"
    sft = {"instruction": "请回答下面的中文医学考试题，并给出简短解析。", "input": prompt_body, "output": chosen}
    dpo = {"prompt": f"请回答下面的中文医学考试题，并给出简短解析。\n{prompt_body}", "chosen": chosen, "rejected": rejected}
    return sft, dpo


def read_cmb_baseline(max_train: int = 2000) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    zip_path = Path("baselines/CMB/data/CMB.zip")
    sft_rows: list[dict[str, str]] = []
    dpo_rows: list[dict[str, str]] = []
    if not zip_path.exists():
        return sft_rows, dpo_rows
    targets = [
        ("CMB/CMB-Exam/CMB-train/CMB-train-merge.json", max_train),
        ("CMB/CMB-Exam/CMB-val/CMB-val-merge.json", 500),
    ]
    with zipfile.ZipFile(zip_path) as zf:
        for name, limit in targets:
            rows = json.loads(zf.read(name).decode("utf-8"))
            for row in rows[:limit]:
                sft, dpo = cmb_to_examples(row)
                if sft:
                    sft_rows.append(sft)
                if dpo:
                    dpo_rows.append(dpo)
        clin = json.loads(zf.read("CMB/CMB-Clin/CMB-Clin-qa.json").decode("utf-8"))
        for case in clin:
            desc = str(case.get("description", "")).strip()
            title = str(case.get("title", "")).strip()
            for qa in case.get("QA_pairs", [])[:6]:
                q = str(qa.get("question", "")).strip()
                a = str(qa.get("answer", "")).strip()
                if q and a:
                    sft_rows.append(
                        {
                            "instruction": "请根据临床病例资料回答医学问题，注意给出谨慎、专业的分析。",
                            "input": f"{title}\n{desc}\n问题：{q}"[:3500],
                            "output": a[:2500],
                        }
                    )
    return sft_rows, dpo_rows


def read_medicalgpt_sft() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = [Path("baselines/MedicalGPT/data/sft/medical_sft_1K_format.jsonl")]
    for path in paths:
        for row in read_jsonl(path):
            conv = row.get("conversations")
            if not isinstance(conv, list):
                continue
            last_user = ""
            for turn in conv:
                role = turn.get("from")
                value = str(turn.get("value", "")).strip()
                if role == "human":
                    last_user = value
                elif role == "gpt" and last_user and value:
                    rows.append(
                        {
                            "instruction": "请作为中文医疗助手回答用户问题，保持谨慎，不替代线下诊断。",
                            "input": last_user[:1800],
                            "output": value[:2200],
                        }
                    )
                    last_user = ""
    return rows


def read_medicalgpt_dpo_medical_only() -> list[dict[str, str]]:
    path = Path("baselines/MedicalGPT/data/reward/dpo_zh_500.jsonl")
    keywords = ["症", "病", "药", "医", "痛", "治疗", "检查", "诊断", "医院", "患者"]
    rows: list[dict[str, str]] = []
    for row in read_jsonl(path):
        conv = row.get("conversations")
        prompt = ""
        if isinstance(conv, list) and conv:
            prompt = str(conv[0].get("value", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        text = prompt + chosen + rejected
        if prompt and chosen and rejected and any(k in text for k in keywords):
            rows.append({"prompt": prompt[:2000], "chosen": chosen[:2400], "rejected": rejected[:2400]})
    return rows


def generic_med_sft(row: dict[str, Any]) -> dict[str, str] | None:
    instruction = first_text(row, ["instruction", "question", "query", "prompt", "input"])
    inp = first_text(row, ["input", "context", "history"])
    output = first_text(row, ["output", "answer", "response", "target"])
    if not instruction or not output:
        text = first_text(row, ["text", "content", "conversation", "conversations", "messages"])
        if text and len(text) > 20:
            return {
                "instruction": "请根据以下医疗对话或医学资料，给出专业、谨慎、易懂的回答。",
                "input": text[:3000],
                "output": "请结合症状、风险和就医建议进行回答。",
            }
        return None
    return {"instruction": instruction[:1200], "input": inp[:1800], "output": output[:2000]}


def redflag_examples() -> list[dict[str, Any]]:
    cases = [
        {
            "symptom": "患者突发胸痛，伴大汗、恶心，持续20分钟不缓解。",
            "risk": "emergency",
            "key": ["立即停止活动", "尽快拨打急救电话", "前往急诊评估心梗风险"],
            "avoid": ["不要自行开车去医院", "不要因症状缓解就继续观察"],
        },
        {
            "symptom": "老人突然口角歪斜、说话含糊，一侧肢体无力。",
            "risk": "emergency",
            "key": ["按卒中FAST识别", "记录发病时间", "立即呼叫急救"],
            "avoid": ["不要自行服用降压药等待", "不要按摩或针刺延误就医"],
        },
        {
            "symptom": "误服大量不明药片后嗜睡、意识模糊。",
            "risk": "emergency",
            "key": ["立即联系急救或中毒控制中心", "保留药品包装", "监测呼吸和意识"],
            "avoid": ["不要自行催吐", "不要强行喂水或喂食"],
        },
        {
            "symptom": "进食海鲜后全身风团、嘴唇肿胀、喘不过气。",
            "risk": "emergency",
            "key": ["警惕严重过敏反应", "立即急救", "如已有肾上腺素笔按医嘱使用"],
            "avoid": ["不要只服用普通抗过敏药后继续等待", "不要平躺导致呼吸更困难"],
        },
        {
            "symptom": "儿童高热伴抽搐，抽搐后意识恢复慢。",
            "risk": "emergency",
            "key": ["保持侧卧防误吸", "记录抽搐持续时间", "及时就医"],
            "avoid": ["不要往嘴里塞东西", "不要强行按压四肢"],
        },
    ]
    rows = []
    for case in cases:
        prompt = f"患者情况：{case['symptom']}\n请判断风险等级并给出安全分诊建议，必须用JSON输出。"
        output = {
            "risk_level": case["risk"],
            "possible_reason": "存在需要立即排查的红旗症状，线上建议不能替代急诊评估。",
            "key_actions": case["key"],
            "avoid_actions": case["avoid"],
            "disclaimer": "本回答仅供健康科普和分诊参考，不能替代医生诊断。"
        }
        rejected = {
            "risk_level": "routine",
            "possible_reason": "可以先在家观察。",
            "key_actions": ["多喝水", "休息"],
            "avoid_actions": [],
            "disclaimer": ""
        }
        rows.append(
            {
                "instruction": "你是谨慎的中文医疗分诊助手。请识别红旗症状，并输出结构化安全建议。",
                "input": f"患者情况：{case['symptom']}",
                "output": json.dumps(output, ensure_ascii=False),
                "prompt": prompt,
                "chosen": json.dumps(output, ensure_ascii=False),
                "rejected": json.dumps(rejected, ensure_ascii=False),
                "key_actions": case["key"],
                "avoid_actions": case["avoid"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    args = parser.parse_args()

    random.seed(42)
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    samples_dir = Path(manifest["paths"]["samples_dir"])
    processed_dir = Path(manifest["paths"]["processed_dir"])
    lf_dir = Path(manifest["paths"]["llamafactory_data_dir"])

    sft_rows: list[dict[str, str]] = []
    dpo_rows: list[dict[str, str]] = []

    cmb_sft, cmb_dpo = read_cmb_baseline(max_train=2000)
    sft_rows.extend(cmb_sft)
    dpo_rows.extend(cmb_dpo)
    sft_rows.extend(read_medicalgpt_sft())
    dpo_rows.extend(read_medicalgpt_dpo_medical_only())

    for sample_path in samples_dir.glob("*.jsonl"):
        rows = read_jsonl(sample_path)
        for row in rows:
            if "ceval_med" in sample_path.name:
                sft, dpo = ceval_to_examples(row)
                if sft:
                    sft_rows.append(sft)
                if dpo:
                    dpo_rows.append(dpo)
            else:
                sft = generic_med_sft(row)
                if sft:
                    sft_rows.append(sft)

    redflag = redflag_examples()
    redflag_sft = [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in redflag]
    redflag_dpo = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in redflag]
    dpo_rows.extend(redflag_dpo)

    if not sft_rows:
        sft_rows.extend(redflag_sft)

    mpo_rows = []
    for row in dpo_rows:
        item = dict(row)
        item["safety_weight"] = 1.0 if "risk_level" in row.get("chosen", "") else 0.3
        item["sft_weight"] = 0.25
        item["preference_weight"] = 1.0
        mpo_rows.append(item)

    write_jsonl(processed_dir / "med_sft_seed.jsonl", sft_rows)
    write_jsonl(processed_dir / "redflag_sft_seed.jsonl", redflag_sft)
    write_jsonl(processed_dir / "med_dpo_seed.jsonl", dpo_rows)
    write_jsonl(processed_dir / "med_mpo_seed.jsonl", mpo_rows)
    write_jsonl(processed_dir / "redflag_triage_seed.jsonl", redflag)

    lf_dir.mkdir(parents=True, exist_ok=True)
    for name in ["med_sft_seed", "redflag_sft_seed", "med_dpo_seed", "med_mpo_seed"]:
        src = processed_dir / f"{name}.jsonl"
        dst = lf_dir / f"{name}.jsonl"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    dataset_info = {
        "med_sft_seed": {
            "file_name": "med_sft_seed.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "redflag_sft_seed": {
            "file_name": "redflag_sft_seed.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "med_dpo_seed": {
            "file_name": "med_dpo_seed.jsonl",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
        "med_mpo_seed": {
            "file_name": "med_mpo_seed.jsonl",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    write_json(lf_dir / "dataset_info.json", dataset_info)
    summary = {
        "sft": len(sft_rows),
        "redflag_sft": len(redflag_sft),
        "dpo": len(dpo_rows),
        "mpo": len(mpo_rows),
        "sources": {
            "cmb_sft": len(cmb_sft),
            "cmb_dpo": len(cmb_dpo),
            "medicalgpt_sft": len(read_medicalgpt_sft()),
            "redflag_manual": len(redflag),
        },
    }
    write_json(processed_dir / "seed_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
