from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def format_options(option: Any) -> str:
    if not isinstance(option, dict):
        return ""
    lines = []
    for key in sorted(option):
        value = safe_text(option.get(key), 500)
        if value:
            lines.append(f"{key}. {value}")
    return "\n".join(lines)


def cmb_prompt(row: dict[str, Any]) -> str:
    meta = " / ".join(
        safe_text(row.get(k), 80)
        for k in ["exam_type", "exam_class", "exam_subject", "question_type"]
        if row.get(k)
    )
    question = safe_text(row.get("question"), 1500)
    options = format_options(row.get("option"))
    parts = []
    if meta:
        parts.append(f"题目来源：{meta}")
    parts.append(question)
    if options:
        parts.append(options)
    return "\n".join(parts)


def cmb_to_sft(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = cmb_prompt(row)
    answer = safe_text(row.get("answer"), 64)
    if not prompt or not answer:
        return None
    explanation = safe_text(row.get("explanation"), 1800)
    if explanation:
        output = f"答案：{answer}\n解析：{explanation}"
    else:
        output = f"答案：{answer}\n解析：根据题干、选项和医学知识，最符合要求的答案是 {answer}。"
    return {
        "instruction": "请回答下面的中文医学考试题，并给出简短解析。",
        "input": prompt,
        "output": output,
        "source": "CMB-Exam",
        "subject": safe_text(row.get("exam_subject"), 80),
    }


def cmb_to_dpo(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = cmb_prompt(row)
    answer = safe_text(row.get("answer"), 64)
    option = row.get("option") if isinstance(row.get("option"), dict) else {}
    if not prompt or not answer or not option:
        return None
    answer_set = set(answer)
    wrongs = [k for k in sorted(option) if k not in answer_set]
    if not wrongs:
        return None
    wrong = wrongs[0]
    explanation = safe_text(row.get("explanation"), 1600)
    chosen_reason = explanation if explanation else f"该题正确答案为 {answer}，需要结合题干关键信息和医学知识判断。"
    rejected_reason = f"选择 {wrong} 与题干要求不匹配，可能遗漏关键医学条件或混淆相近概念。"
    return {
        "prompt": f"请回答下面的中文医学考试题，并给出简短解析。\n{prompt}",
        "chosen": f"答案：{answer}\n解析：{chosen_reason}",
        "rejected": f"答案：{wrong}\n解析：{rejected_reason}",
        "source": "CMB-Exam",
        "subject": safe_text(row.get("exam_subject"), 80),
    }


def cmb_clin_to_sft(case: dict[str, Any]) -> list[dict[str, str]]:
    title = safe_text(case.get("title"), 200)
    desc = safe_text(case.get("description"), 3200)
    rows = []
    for qa in case.get("QA_pairs", []):
        q = safe_text(qa.get("question"), 600)
        a = safe_text(qa.get("answer"), 2400)
        if q and a:
            rows.append(
                {
                    "instruction": "请根据临床病例资料回答医学问题，注意给出谨慎、专业的分析。",
                    "input": f"{title}\n{desc}\n问题：{q}"[:4000],
                    "output": a,
                    "source": "CMB-Clin",
                    "subject": title,
                }
            )
    return rows


def load_cmb() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    zip_path = Path("baselines/CMB/data/CMB.zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Clone CMB baseline first.")
    with zipfile.ZipFile(zip_path) as zf:
        train = json.loads(zf.read("CMB/CMB-Exam/CMB-train/CMB-train-merge.json").decode("utf-8"))
        val = json.loads(zf.read("CMB/CMB-Exam/CMB-val/CMB-val-merge.json").decode("utf-8"))
        clin = json.loads(zf.read("CMB/CMB-Clin/CMB-Clin-qa.json").decode("utf-8"))
    return train, val, clin


def load_medicalgpt_sft() -> list[dict[str, str]]:
    path = Path("baselines/MedicalGPT/data/sft/medical_sft_1K_format.jsonl")
    out = []
    for row in read_jsonl(path):
        conv = row.get("conversations")
        if not isinstance(conv, list):
            continue
        last_user = ""
        for turn in conv:
            role = turn.get("from")
            text = safe_text(turn.get("value"), 2400)
            if role == "human":
                last_user = text
            elif role == "gpt" and last_user and text:
                out.append(
                    {
                        "instruction": "请作为中文医疗助手回答用户问题，保持谨慎，不替代线下诊断。",
                        "input": last_user,
                        "output": text,
                        "source": "MedicalGPT",
                        "subject": "medical_dialogue",
                    }
                )
                last_user = ""
    return out


def load_medicalgpt_dpo() -> list[dict[str, str]]:
    path = Path("baselines/MedicalGPT/data/reward/dpo_zh_500.jsonl")
    keywords = ["症", "病", "药", "医", "痛", "治疗", "检查", "诊断", "医院", "患者", "发热", "咳嗽"]
    out = []
    for row in read_jsonl(path):
        conv = row.get("conversations")
        prompt = ""
        if isinstance(conv, list) and conv:
            prompt = safe_text(conv[0].get("value"), 2000)
        chosen = safe_text(row.get("chosen"), 2600)
        rejected = safe_text(row.get("rejected"), 2600)
        blob = prompt + chosen + rejected
        if prompt and chosen and rejected and any(k in blob for k in keywords):
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "MedicalGPT-DPO", "subject": "mixed"})
    return out


def redflag_templates() -> list[dict[str, Any]]:
    base = [
        ("胸痛", "患者突发压榨样胸痛，伴大汗、恶心，持续20分钟不缓解。", ["立即停止活动", "尽快拨打急救电话", "前往急诊评估心梗风险"], ["不要自行开车去医院", "不要因症状缓解就继续观察"]),
        ("卒中", "老人突然口角歪斜、说话含糊，一侧肢体无力。", ["按卒中FAST识别", "记录发病时间", "立即呼叫急救"], ["不要自行服用降压药等待", "不要按摩或针刺延误就医"]),
        ("中毒", "误服大量不明药片后嗜睡、意识模糊。", ["立即联系急救或中毒控制中心", "保留药品包装", "监测呼吸和意识"], ["不要自行催吐", "不要强行喂水或喂食"]),
        ("过敏", "进食海鲜后全身风团、嘴唇肿胀、喘不过气。", ["警惕严重过敏反应", "立即急救", "如已有肾上腺素笔按医嘱使用"], ["不要只服普通抗过敏药后继续等待", "不要平躺导致呼吸更困难"]),
        ("儿童惊厥", "儿童高热伴抽搐，抽搐后意识恢复慢。", ["保持侧卧防误吸", "记录抽搐持续时间", "及时就医"], ["不要往嘴里塞东西", "不要强行按压四肢"]),
        ("呼吸困难", "患者突然呼吸困难、口唇发紫，说话断续。", ["立即急救", "保持气道通畅", "避免剧烈活动"], ["不要自行加大镇静药剂量", "不要独自等待观察"]),
        ("严重出血", "外伤后伤口喷射样出血，按压后仍持续出血。", ["持续直接压迫止血", "立即呼叫急救", "抬高患肢并保暖"], ["不要频繁掀开敷料查看", "不要用脏布直接填塞深伤口"]),
        ("孕产急症", "孕晚期出现阴道大量出血并伴腹痛。", ["立即就医或呼叫急救", "左侧卧位等待救援", "记录出血量和胎动"], ["不要自行服止痛药", "不要拖延到门诊排队"]),
    ]
    rows = []
    styles = [
        "请用JSON输出风险等级、可能原因、key_actions、avoid_actions和免责声明。",
        "请给出面向家属的分诊建议，必须强调不能替代医生诊断。",
        "请判断是否需要急诊，并列出必须做和必须避免的动作。",
    ]
    for category, symptom, key, avoid in base:
        for style in styles:
            prompt = f"患者情况：{symptom}\n{style}"
            chosen_obj = {
                "risk_level": "emergency",
                "possible_reason": f"存在{category}相关红旗症状，需要尽快线下评估。",
                "key_actions": key,
                "avoid_actions": avoid,
                "disclaimer": "本回答仅供健康科普和分诊参考，不能替代医生诊断。"
            }
            rejected_obj = {
                "risk_level": "routine",
                "possible_reason": "可以先在家观察。",
                "key_actions": ["多喝水", "休息", "自行服药观察"],
                "avoid_actions": [],
                "disclaimer": ""
            }
            rows.append(
                {
                    "instruction": "你是谨慎的中文医疗分诊助手。请识别红旗症状，并输出结构化安全建议。",
                    "input": f"患者情况：{symptom}\n{style}",
                    "output": json.dumps(chosen_obj, ensure_ascii=False),
                    "prompt": prompt,
                    "chosen": json.dumps(chosen_obj, ensure_ascii=False),
                    "rejected": json.dumps(rejected_obj, ensure_ascii=False),
                    "source": "manual-redflag",
                    "subject": category,
                    "key_actions": key,
                    "avoid_actions": avoid,
                }
            )
    return rows


def sample_balanced(rows: list[dict[str, Any]], max_rows: int, key: str, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    keys = list(buckets)
    while len(selected) < max_rows and keys:
        rng.shuffle(keys)
        progressed = False
        for k in list(keys):
            if buckets[k]:
                selected.append(buckets[k].pop())
                progressed = True
                if len(selected) >= max_rows:
                    break
            else:
                keys.remove(k)
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def strip_meta(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: row[k] for k in keys if k in row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/stage1")
    parser.add_argument("--sft-size", type=int, default=30000)
    parser.add_argument("--dpo-size", type=int, default=25000)
    parser.add_argument("--prefix", default="stage1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    lf_dir = Path("data/llamafactory")
    processed = Path("data/processed")

    cmb_train, cmb_val, cmb_clin = load_cmb()
    cmb_sft = [x for x in (cmb_to_sft(r) for r in cmb_train) if x]
    cmb_dpo = [x for x in (cmb_to_dpo(r) for r in cmb_train) if x]
    val_sft = [x for x in (cmb_to_sft(r) for r in cmb_val) if x]
    val_dpo = [x for x in (cmb_to_dpo(r) for r in cmb_val) if x]
    clin_sft = [row for case in cmb_clin for row in cmb_clin_to_sft(case)]
    med_sft = load_medicalgpt_sft()
    med_dpo = load_medicalgpt_dpo()
    redflag = redflag_templates()

    sft_pool = cmb_sft + med_sft + clin_sft + [strip_meta(r, ["instruction", "input", "output", "source", "subject"]) for r in redflag]
    dpo_pool = cmb_dpo + med_dpo + [strip_meta(r, ["prompt", "chosen", "rejected", "source", "subject"]) for r in redflag]

    sft_selected = sample_balanced(sft_pool, args.sft_size, "subject", rng)
    dpo_selected = sample_balanced(dpo_pool, args.dpo_size, "subject", rng)
    sft_val = sample_balanced(val_sft + clin_sft[:120], 500, "subject", rng)
    dpo_val = sample_balanced(val_dpo + [strip_meta(r, ["prompt", "chosen", "rejected", "source", "subject"]) for r in redflag], 500, "subject", rng)
    redflag_sft = [strip_meta(r, ["instruction", "input", "output", "source", "subject"]) for r in redflag]
    redflag_dpo = [strip_meta(r, ["prompt", "chosen", "rejected", "source", "subject"]) for r in redflag]
    mpo_train = []
    for row in dpo_selected:
        item = dict(row)
        item["preference_weight"] = 1.0
        item["sft_weight"] = 0.25
        item["safety_weight"] = 1.0 if row.get("source") == "manual-redflag" else 0.2
        mpo_train.append(item)

    p = args.prefix
    files = {
        f"{p}_med_sft_train.jsonl": sft_selected,
        f"{p}_med_sft_val.jsonl": sft_val,
        f"{p}_med_dpo_train.jsonl": dpo_selected,
        f"{p}_med_dpo_val.jsonl": dpo_val,
        f"{p}_med_mpo_train.jsonl": mpo_train,
        f"{p}_redflag_sft.jsonl": redflag_sft,
        f"{p}_redflag_dpo.jsonl": redflag_dpo,
    }
    for name, rows in files.items():
        write_jsonl(out / name, rows)
        write_jsonl(lf_dir / name, rows)
        write_jsonl(processed / name, rows)

    dataset_info_path = lf_dir / "dataset_info.json"
    dataset_info = {}
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_info.update(
        {
            f"{p}_med_sft_train": {
                "file_name": f"{p}_med_sft_train.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            },
            f"{p}_med_sft_val": {
                "file_name": f"{p}_med_sft_val.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            },
            f"{p}_med_dpo_train": {
                "file_name": f"{p}_med_dpo_train.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            f"{p}_med_dpo_val": {
                "file_name": f"{p}_med_dpo_val.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            f"{p}_med_mpo_train": {
                "file_name": f"{p}_med_mpo_train.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            f"{p}_redflag_sft": {
                "file_name": f"{p}_redflag_sft.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            },
            f"{p}_redflag_dpo": {
                "file_name": f"{p}_redflag_dpo.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
        }
    )
    write_json(dataset_info_path, dataset_info)

    summary = {
        "sft_train": len(sft_selected),
        "sft_val": len(sft_val),
        "dpo_train": len(dpo_selected),
        "dpo_val": len(dpo_val),
        "mpo_train": len(mpo_train),
        "redflag_sft": len(redflag_sft),
        "redflag_dpo": len(redflag_dpo),
        "source_counts_sft": Counter(r.get("source", "unknown") for r in sft_selected),
        "source_counts_dpo": Counter(r.get("source", "unknown") for r in dpo_selected),
        "subject_top10_sft": Counter(r.get("subject", "unknown") for r in sft_selected).most_common(10),
        "subject_top10_dpo": Counter(r.get("subject", "unknown") for r in dpo_selected).most_common(10),
        "input_baseline_sizes": {
            "cmb_train": len(cmb_train),
            "cmb_val": len(cmb_val),
            "cmb_clin_cases": len(cmb_clin),
            "medicalgpt_sft": len(med_sft),
            "medicalgpt_medical_dpo": len(med_dpo),
        },
    }
    write_json(out / f"{p}_data_summary.json", summary)
    write_json(processed / f"{p}_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
