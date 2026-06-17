from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path("data/raw/hf_extra")


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
    instruction = clean_text(instruction, 1400)
    input_text = clean_text(input_text, 3200)
    output = clean_text(output, 4200)
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": input_text, "output": output, "source": source, "subject": clean_text(subject, 160)}


def dpo_row(prompt: str, chosen: str, rejected: str, source: str, subject: str = "") -> dict[str, str] | None:
    prompt = clean_text(prompt, 3200)
    chosen = clean_text(chosen, 4200)
    rejected = clean_text(rejected, 4200)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source, "subject": clean_text(subject, 160)}


def convert_mxode_sft(writer: JsonlWriter) -> int:
    path = ROOT / "Mxode__Chinese-Medical-Instruct-1M" / "medical-train.jsonl"
    if not path.exists():
        return 0
    n = 0
    for item in iter_jsonl(path):
        row = sft_row(
            "请作为中文医疗助手回答用户问题，保持专业、谨慎，不替代线下诊断。",
            item.get("prompt"),
            item.get("response"),
            "Chinese-Medical-Instruct-1M",
        )
        if row:
            writer.write(row)
            n += 1
    return n


def convert_medical_o1(writer: JsonlWriter) -> int:
    root = ROOT / "FreedomIntelligence__medical-o1-reasoning-SFT"
    paths = [root / "medical_o1_sft_Chinese.json", root / "medical_o1_sft_mix_Chinese.json"]
    seen = set()
    n = 0
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            q = clean_text(item.get("Question"), 2400)
            cot = clean_text(item.get("Complex_CoT"), 3200)
            resp = clean_text(item.get("Response"), 2400)
            if not q or not resp:
                continue
            key = (q, resp)
            if key in seen:
                continue
            seen.add(key)
            output = f"分析：{cot}\n结论：{resp}" if cot else resp
            row = sft_row("请回答下面的医学推理问题，并给出分析和结论。", q, output, "medical-o1-reasoning-SFT")
            if row:
                writer.write(row)
                n += 1
    return n


def format_cmexam_prompt(item: dict[str, Any]) -> str:
    opts = item.get("Options")
    lines = []
    if isinstance(opts, list):
        for opt in opts:
            key = opt.get("key", "")
            value = opt.get("value", "")
            if key and value:
                lines.append(f"{key}. {clean_text(value, 500)}")
    return clean_text(item.get("Question"), 1800) + ("\n" + "\n".join(lines) if lines else "")


def convert_cmexam(sft_writer: JsonlWriter, dpo_writer: JsonlWriter, eval_writer: JsonlWriter) -> dict[str, int]:
    root = ROOT / "fzkuji__CMExam"
    counts = Counter()
    for split in ["train", "valid", "test"]:
        path = root / f"{split}.json"
        if not path.exists():
            continue
        for idx, item in enumerate(iter_jsonl(path), start=1):
            prompt = format_cmexam_prompt(item)
            answer = clean_text(item.get("Answer"), 80)
            explanation = clean_text(item.get("Explanation"), 2600)
            if not prompt or not answer:
                continue
            chosen = f"答案：{answer}\n解析：{explanation or '根据题干与选项，正确答案如上。'}"
            row = sft_row("请回答下面的中文医学考试题，并给出解析。", prompt, chosen, "CMExam", split)
            if row and split == "train":
                sft_writer.write(row)
                counts["sft_train"] += 1
            wrong = None
            opts = item.get("Options")
            if isinstance(opts, list):
                for opt in opts:
                    k = str(opt.get("key", ""))
                    if k and k not in set(answer):
                        wrong = k
                        break
            if wrong and split == "train":
                rejected = f"答案：{wrong}\n解析：该答案与题干医学知识点不匹配，存在误导风险。"
                pref = dpo_row(f"请回答下面的中文医学考试题，并给出解析。\n{prompt}", chosen, rejected, "CMExam", split)
                if pref:
                    dpo_writer.write(pref)
                    counts["dpo_train"] += 1
            eval_writer.write(
                {
                    "id": f"CMExam-{split}-{idx}",
                    "prompt": prompt,
                    "answer": answer,
                    "explanation": explanation,
                    "source": "CMExam",
                    "split": split,
                }
            )
            counts[f"eval_{split}"] += 1
    return dict(counts)


def convert_chatdoctor(writer: JsonlWriter) -> int:
    path = ROOT / "lavita__ChatDoctor-HealthCareMagic-100k" / "data" / "train-00000-of-00001-5e7cb295b9cff0bf.parquet"
    if not path.exists():
        return 0
    n = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=5000):
        for item in batch.to_pylist():
            row = sft_row(item.get("instruction"), item.get("input"), item.get("output"), "ChatDoctor-HealthCareMagic-100k", "english")
            if row:
                writer.write(row)
                n += 1
    return n


def convert_ai_medical_pretrain(writer: JsonlWriter) -> int:
    root = ROOT / "ruslanmv__ai-medical-dataset" / "data"
    n = 0
    for path in sorted(root.glob("train-*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=10000):
            for item in batch.to_pylist():
                q = clean_text(item.get("question"), 1000)
                ctx = clean_text(item.get("context"), 5000)
                if not q and not ctx:
                    continue
                text = f"Question: {q}\nMedical context: {ctx}" if q else ctx
                writer.write({"text": text, "source": "ai-medical-dataset", "subject": "english-medical-corpus"})
                n += 1
    return n


def register_dataset_info(dataset_info_path: Path) -> None:
    info = {}
    if dataset_info_path.exists():
        info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "extra_med_sft_full": {
                "file_name": "extra_med_sft_full.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            },
            "extra_med_dpo_full": {
                "file_name": "extra_med_dpo_full.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            "extra_med_mpo_full": {
                "file_name": "extra_med_mpo_full.jsonl",
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            },
            "extra_med_pretrain_full": {
                "file_name": "extra_med_pretrain_full.jsonl",
                "columns": {"prompt": "text"},
            },
        }
    )
    write_json(dataset_info_path, info)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/extra_processed")
    parser.add_argument("--llamafactory-dir", default="data/llamafactory")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    lf_dir = Path(args.llamafactory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lf_dir.mkdir(parents=True, exist_ok=True)

    sft = JsonlWriter(out_dir / "extra_med_sft_full.jsonl")
    dpo = JsonlWriter(out_dir / "extra_med_dpo_full.jsonl")
    eval_writer = JsonlWriter(Path("data/eval/extra_cmexam_eval.jsonl"))
    counts: dict[str, Any] = {}
    try:
        counts["mxode_chinese_medical_instruct_1m"] = convert_mxode_sft(sft)
        counts["medical_o1_reasoning"] = convert_medical_o1(sft)
        counts["chatdoctor_en_sft"] = convert_chatdoctor(sft)
        counts["cmexam"] = convert_cmexam(sft, dpo, eval_writer)
    finally:
        sft.close()
        dpo.close()
        eval_writer.close()

    mpo_path = out_dir / "extra_med_mpo_full.jsonl"
    with (out_dir / "extra_med_dpo_full.jsonl").open("r", encoding="utf-8") as src, mpo_path.open("w", encoding="utf-8") as dst:
        mpo_count = 0
        for line in src:
            item = json.loads(line)
            item["preference_weight"] = 1.0
            item["sft_weight"] = 0.2
            item["safety_weight"] = 0.3
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            mpo_count += 1
    counts["extra_mpo"] = mpo_count

    pretrain = JsonlWriter(out_dir / "extra_med_pretrain_full.jsonl")
    try:
        counts["ai_medical_pretrain"] = convert_ai_medical_pretrain(pretrain)
    finally:
        pretrain.close()

    for name in ["extra_med_sft_full.jsonl", "extra_med_dpo_full.jsonl", "extra_med_mpo_full.jsonl", "extra_med_pretrain_full.jsonl"]:
        hardlink_or_copy(out_dir / name, lf_dir / name)
    register_dataset_info(lf_dir / "dataset_info.json")

    summary = {
        "counts_by_converter": counts,
        "sft_rows": sft.count,
        "sft_source_counts": dict(sft.sources),
        "dpo_rows": dpo.count,
        "dpo_source_counts": dict(dpo.sources),
        "mpo_rows": mpo_count,
        "pretrain_rows": pretrain.count,
        "pretrain_source_counts": dict(pretrain.sources),
        "cmexam_eval_rows": eval_writer.count,
        "files": {
            p.name: p.stat().st_size
            for p in [
                out_dir / "extra_med_sft_full.jsonl",
                out_dir / "extra_med_dpo_full.jsonl",
                out_dir / "extra_med_mpo_full.jsonl",
                out_dir / "extra_med_pretrain_full.jsonl",
            ]
        },
    }
    write_json(out_dir / "extra_processed_summary.json", summary)
    write_json(Path("data/processed/extra_processed_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
