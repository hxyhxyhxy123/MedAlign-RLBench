from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("data/raw/hf")


def clean_text(value: Any, limit: int = 6000) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u0000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("w", encoding="utf-8")
        self.count = 0
        self.sources = Counter()

    def write(self, row: dict[str, Any]) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1
        self.sources[str(row.get("source", "unknown"))] += 1

    def close(self) -> None:
        self.f.close()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sft_row(instruction: str, input_text: str, output: str, source: str, subject: str = "") -> dict[str, str] | None:
    instruction = clean_text(instruction, 1200)
    input_text = clean_text(input_text, 3000)
    output = clean_text(output, 4000)
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": input_text, "output": output, "source": source, "subject": clean_text(subject, 120)}


def dpo_row(prompt: str, chosen: str, rejected: str, source: str, subject: str = "") -> dict[str, str] | None:
    prompt = clean_text(prompt, 3000)
    chosen = clean_text(chosen, 4000)
    rejected = clean_text(rejected, 4000)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source, "subject": clean_text(subject, 120)}


def write_conversation_pairs(writer: JsonlWriter, conversation: list[dict[str, Any]], source: str, subject: str = "") -> int:
    written = 0
    pending_user = ""
    for turn in conversation:
        role = turn.get("from", turn.get("role", ""))
        content = clean_text(turn.get("value", turn.get("content", "")), 3500)
        if role in {"human", "user"}:
            pending_user = content
        elif role in {"gpt", "assistant"} and pending_user and content:
            row = sft_row("请作为中文医疗助手回答用户问题，保持专业、谨慎，不替代线下诊断。", pending_user, content, source, subject)
            if row:
                writer.write(row)
                written += 1
            pending_user = ""
    return written


def convert_huatuogpt2(writer: JsonlWriter) -> int:
    path = ROOT / "FreedomIntelligence__HuatuoGPT2-SFT-GPT4-140K" / "HuatuoGPT2-GPT4-SFT-140K.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for item in data:
        conv = item.get("conversations")
        if isinstance(conv, list):
            n += write_conversation_pairs(writer, conv, "HuatuoGPT2-SFT-GPT4-140K")
    return n


def convert_huatuogpt_v1(writer: JsonlWriter) -> int:
    path = ROOT / "FreedomIntelligence__HuatuoGPT-sft-data-v1" / "HuatuoGPT_sft_data_v1.jsonl"
    if not path.exists():
        return 0
    n = 0
    for item in iter_jsonl(path):
        data = item.get("data")
        if not isinstance(data, list) or len(data) < 2:
            continue
        question = clean_text(data[0]).removeprefix("问：")
        answer = clean_text(data[1]).removeprefix("答：")
        row = sft_row("请作为中文医疗助手回答用户问题。", question, answer, "HuatuoGPT-sft-data-v1")
        if row:
            writer.write(row)
            n += 1
    return n


def convert_huatuo26m_lite(writer: JsonlWriter) -> int:
    path = ROOT / "FreedomIntelligence__Huatuo26M-Lite" / "format_data.jsonl"
    if not path.exists():
        return 0
    n = 0
    for item in iter_jsonl(path):
        row = sft_row(
            "请作为中文医疗助手回答用户问题，回答应安全、清晰、避免过度诊断。",
            item.get("question"),
            item.get("answer"),
            "Huatuo26M-Lite",
            item.get("label", ""),
        )
        if row:
            writer.write(row)
            n += 1
    return n


def convert_disc_med(writer: JsonlWriter) -> int:
    path = ROOT / "Flmc__DISC-Med-SFT" / "DISC-Med-SFT_released.jsonl"
    if not path.exists():
        return 0
    n = 0
    for item in iter_jsonl(path):
        conv = item.get("conversation")
        if isinstance(conv, list):
            n += write_conversation_pairs(writer, conv, "DISC-Med-SFT", item.get("source", ""))
    return n


def convert_shibing_finetune(writer: JsonlWriter) -> int:
    paths = [
        ROOT / "shibing624__medical" / "finetune" / "train_zh_0.json",
        ROOT / "shibing624__medical" / "finetune" / "valid_zh_0.json",
        ROOT / "shibing624__medical" / "finetune" / "test_zh_0.json",
    ]
    n = 0
    for path in paths:
        if not path.exists():
            continue
        for item in iter_jsonl(path):
            row = sft_row(
                item.get("instruction", "请回答下面的医学问题。"),
                item.get("input", ""),
                item.get("output", ""),
                "shibing624-medical-finetune",
                path.name,
            )
            if row:
                writer.write(row)
                n += 1
    return n


def convert_shibing_reward(writer: JsonlWriter) -> int:
    paths = [
        ROOT / "shibing624__medical" / "reward" / "train.json",
        ROOT / "shibing624__medical" / "reward" / "valid.json",
        ROOT / "shibing624__medical" / "reward" / "test.json",
    ]
    n = 0
    for path in paths:
        if not path.exists():
            continue
        for item in iter_jsonl(path):
            row = dpo_row(
                item.get("question"),
                item.get("response_chosen"),
                item.get("response_rejected"),
                "shibing624-medical-reward",
                path.name,
            )
            if row:
                writer.write(row)
                n += 1
    return n


def convert_pretrain(writer: JsonlWriter) -> int:
    paths = [
        ROOT / "shibing624__medical" / "pretrain" / "medical_book_zh.json",
        ROOT / "shibing624__medical" / "pretrain" / "train_encyclopedia.json",
        ROOT / "shibing624__medical" / "pretrain" / "valid_encyclopedia.json",
        ROOT / "shibing624__medical" / "pretrain" / "test_encyclopedia.json",
    ]
    n = 0
    for path in paths:
        if not path.exists():
            continue
        for item in iter_jsonl(path):
            text = clean_text(item.get("text"), 6000)
            if text:
                writer.write({"text": text, "source": "shibing624-medical-pretrain", "subject": path.name})
                n += 1
    return n


def register_dataset_info(dataset_info_path: Path) -> None:
    info = {}
    if dataset_info_path.exists():
        info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "hf_med_sft_full": {
                "file_name": "hf_med_sft_full.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            },
            "hf_med_dpo_full": {
                "file_name": "hf_med_dpo_full.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            "hf_med_mpo_full": {
                "file_name": "hf_med_mpo_full.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            "hf_med_pretrain_full": {
                "file_name": "hf_med_pretrain_full.jsonl",
                "columns": {"prompt": "text"},
            },
        }
    )
    write_json(dataset_info_path, info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/hf_processed")
    parser.add_argument("--llamafactory-dir", default="data/llamafactory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lf_dir = Path(args.llamafactory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lf_dir.mkdir(parents=True, exist_ok=True)

    sft_writer = JsonlWriter(out_dir / "hf_med_sft_full.jsonl")
    counts = {}
    try:
        counts["huatuogpt2_sft"] = convert_huatuogpt2(sft_writer)
        counts["huatuogpt_v1"] = convert_huatuogpt_v1(sft_writer)
        counts["huatuo26m_lite"] = convert_huatuo26m_lite(sft_writer)
        counts["disc_med_sft"] = convert_disc_med(sft_writer)
        counts["shibing624_finetune"] = convert_shibing_finetune(sft_writer)
    finally:
        sft_writer.close()

    dpo_writer = JsonlWriter(out_dir / "hf_med_dpo_full.jsonl")
    try:
        counts["shibing624_reward_dpo"] = convert_shibing_reward(dpo_writer)
    finally:
        dpo_writer.close()

    mpo_path = out_dir / "hf_med_mpo_full.jsonl"
    with (out_dir / "hf_med_dpo_full.jsonl").open("r", encoding="utf-8") as src, mpo_path.open("w", encoding="utf-8") as dst:
        mpo_count = 0
        for line in src:
            item = json.loads(line)
            item["preference_weight"] = 1.0
            item["sft_weight"] = 0.25
            item["safety_weight"] = 0.5
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            mpo_count += 1
    counts["hf_mpo"] = mpo_count

    pretrain_writer = JsonlWriter(out_dir / "hf_med_pretrain_full.jsonl")
    try:
        counts["shibing624_pretrain"] = convert_pretrain(pretrain_writer)
    finally:
        pretrain_writer.close()

    for name in ["hf_med_sft_full.jsonl", "hf_med_dpo_full.jsonl", "hf_med_mpo_full.jsonl", "hf_med_pretrain_full.jsonl"]:
        target = lf_dir / name
        if target.exists():
            target.unlink()
        src = out_dir / name
        try:
            os.link(src, target)
        except OSError:
            shutil.copyfile(src, target)

    register_dataset_info(lf_dir / "dataset_info.json")
    summary = {
        "counts_by_converter": counts,
        "sft_rows": sft_writer.count,
        "sft_source_counts": dict(sft_writer.sources),
        "dpo_rows": dpo_writer.count,
        "dpo_source_counts": dict(dpo_writer.sources),
        "mpo_rows": mpo_count,
        "pretrain_rows": pretrain_writer.count,
        "pretrain_source_counts": dict(pretrain_writer.sources),
        "files": {
            name: (out_dir / name).stat().st_size
            for name in ["hf_med_sft_full.jsonl", "hf_med_dpo_full.jsonl", "hf_med_mpo_full.jsonl", "hf_med_pretrain_full.jsonl"]
        },
    }
    write_json(out_dir / "hf_processed_summary.json", summary)
    write_json(Path("data/processed/hf_processed_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
