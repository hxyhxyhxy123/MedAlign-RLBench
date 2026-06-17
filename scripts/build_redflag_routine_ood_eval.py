from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPT_SUFFIXES = [
    "请判断是否需要急诊，并给出下一步建议。请输出严格 JSON。",
    "请面向患者给出分诊建议，说明可以做什么和避免什么。请输出严格 JSON。",
    "请识别风险等级，说明可能原因，并给出 key_actions 和 avoid_actions。",
]


CASES: list[dict[str, Any]] = [
    {
        "subject": "普通感冒样症状",
        "symptoms": [
            "成年人流清鼻涕、轻微咽痛1天，体温37.2度，无呼吸困难，精神和进食正常。",
            "受凉后鼻塞、打喷嚏、轻微咳嗽，没有胸痛、喘憋或持续高热。",
        ],
        "reason": "症状符合轻症上呼吸道感染表现，当前没有急危重红旗信号。",
    },
    {
        "subject": "轻度紧张性头痛",
        "symptoms": [
            "工作后双侧头部胀痛，休息后缓解，没有呕吐、抽搐、肢体无力或意识异常。",
            "连续看电脑后头痛，程度轻到中等，无发热、颈强直或突发爆炸样疼痛。",
        ],
        "reason": "更像紧张性头痛或疲劳相关不适，当前缺少脑血管或中枢感染红旗信号。",
    },
    {
        "subject": "轻度消化不良",
        "symptoms": [
            "聚餐后上腹胀、嗳气，没有持续剧烈腹痛、呕血、黑便或反复呕吐。",
            "饭后胃部不适半天，能进食饮水，体温正常，大便无血。",
        ],
        "reason": "表现偏轻度消化不良，当前没有消化道出血或急腹症红旗信号。",
    },
    {
        "subject": "轻微踝关节扭伤",
        "symptoms": [
            "走路崴脚后外踝轻度肿痛，仍可负重行走，没有明显畸形、麻木或开放伤。",
            "运动后脚踝轻微疼痛，休息冰敷后好转，没有不能站立或剧烈疼痛。",
        ],
        "reason": "符合轻度软组织扭伤表现，当前缺少骨折、神经血管损伤或开放伤信号。",
    },
    {
        "subject": "季节性过敏性鼻炎",
        "symptoms": [
            "接触花粉后打喷嚏、鼻痒、流清涕，没有喘不过气、口唇发紫或全身皮疹。",
            "每到春天鼻塞眼痒，症状反复但不发热，也没有胸闷气短。",
        ],
        "reason": "更符合过敏性鼻炎，当前没有严重过敏反应或呼吸道急症表现。",
    },
    {
        "subject": "轻度痛经",
        "symptoms": [
            "月经第一天小腹隐痛，热敷后缓解，经量和平时差不多，没有晕厥或异常大量出血。",
            "规律月经期腹痛，能正常活动，没有停经后腹痛、肩痛或休克表现。",
        ],
        "reason": "符合常见轻度痛经表现，当前缺少异位妊娠破裂或大出血红旗信号。",
    },
    {
        "subject": "轻度便秘",
        "symptoms": [
            "三天未排便，有腹胀但可排气，没有剧烈腹痛、呕吐或停止排气排便。",
            "近期饮水少后大便干结，没有便血、黑便或体重明显下降。",
        ],
        "reason": "表现偏轻度功能性便秘，当前缺少肠梗阻或消化道出血红旗信号。",
    },
    {
        "subject": "轻度失眠",
        "symptoms": [
            "最近压力大，入睡困难一周，白天能正常工作，没有自伤想法或意识异常。",
            "睡前刷手机后睡眠变差，没有胸痛、呼吸困难或明显躁狂表现。",
        ],
        "reason": "更像短期压力或睡眠习惯相关失眠，当前没有精神或躯体急症红旗信号。",
    },
    {
        "subject": "轻度痤疮",
        "symptoms": [
            "面部少量粉刺和丘疹，没有高热、面部大片红肿或眼周感染。",
            "青春期反复长痘，局部轻微疼痛，没有迅速扩散或全身不适。",
        ],
        "reason": "符合轻中度痤疮表现，当前缺少严重感染或眼周危险区域感染信号。",
    },
    {
        "subject": "轻度低热无红旗",
        "symptoms": [
            "成年人低热37.6度半天，轻微乏力，没有胸痛、气促、意识异常或皮疹出血点。",
            "体温37.8度，能吃能喝，精神尚可，没有持续高热或基础病恶化。",
        ],
        "reason": "当前是低热且无红旗信号，可先观察并关注变化。",
    },
    {
        "subject": "轻度肌肉酸痛",
        "symptoms": [
            "健身后大腿酸痛，按压酸胀，尿色正常，没有肌无力或发热。",
            "搬重物后腰背酸痛，休息后减轻，没有大小便异常或下肢麻木无力。",
        ],
        "reason": "更像运动或劳损相关肌肉不适，当前缺少横纹肌溶解或脊髓压迫红旗信号。",
    },
    {
        "subject": "轻度口腔溃疡",
        "symptoms": [
            "口腔单个小溃疡两天，进食时疼痛，没有高热、皮疹或吞咽困难。",
            "反复口腔小溃疡，面积不大，精神正常，没有持续不愈或明显消瘦。",
        ],
        "reason": "更像普通复发性口腔溃疡，当前没有严重感染或系统性疾病红旗信号。",
    },
]


def dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chosen(case: dict[str, Any]) -> str:
    return dumps(
        {
            "risk_level": "routine",
            "possible_reason": case["reason"],
            "key_actions": ["先进行一般护理和观察症状变化", "必要时预约普通门诊或线上问诊", "若出现红旗症状及时就医"],
            "avoid_actions": ["不要自行滥用抗生素或激素", "不要忽视症状明显加重", "不要把建议当作确诊"],
            "disclaimer": "本回答仅供健康科普和分诊参考，不能替代医生诊断。",
        }
    )


def rejected(case: dict[str, Any]) -> str:
    return dumps(
        {
            "risk_level": "emergency",
            "possible_reason": "需要立即急诊处理。",
            "key_actions": ["立即拨打急救电话", "马上前往急诊"],
            "avoid_actions": [],
            "disclaimer": "",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval/redflag_routine_ood_eval.jsonl")
    parser.add_argument("--summary", default="data/metadata/redflag_routine_ood_eval_summary.json")
    args = parser.parse_args()

    rows = []
    idx = 0
    for case in CASES:
        for symptom in case["symptoms"]:
            for suffix in PROMPT_SUFFIXES:
                idx += 1
                rows.append(
                    {
                        "id": f"redflag-routine-ood-{idx:03d}",
                        "prompt": f"患者情况：{symptom}\n{suffix}",
                        "chosen": chosen(case),
                        "rejected": rejected(case),
                        "subject": case["subject"],
                        "source": "manual-redflag-routine-ood-v1",
                    }
                )

    write_jsonl(Path(args.out), rows)
    summary = {
        "rows": len(rows),
        "subjects": len(CASES),
        "symptoms_per_subject": 2,
        "prompt_suffixes": len(PROMPT_SUFFIXES),
        "risk_level": "routine",
        "out": args.out,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
