from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def pp(new: float, old: float) -> str:
    return f"{100 * (new - old):+.2f} pp"


def main() -> None:
    cmb_base = load_json("runs/metrics/cmb_val_base.json")
    cmb_sft = load_json("runs/metrics/cmb_val_sft.json")
    cmb_dpo = load_json("runs/metrics/cmb_val_dpo.json")
    cmb_mpo = load_json("runs/metrics/cmb_val_mpo.json")
    ceval_base = load_json("runs/metrics/ceval_med_base.json")
    ceval_sft = load_json("runs/metrics/ceval_med_sft.json")
    ceval_dpo = load_json("runs/metrics/ceval_med_dpo.json")
    red_base = load_json("runs/metrics/redflag_aug_base.json")
    red_sft = load_json("runs/metrics/redflag_aug_sft.json")
    red_dpo = load_json("runs/metrics/redflag_aug_dpo.json")
    red_mpo = load_json("runs/metrics/redflag_aug_mpo.json")
    kvcache = load_json("runs/kvcache_benchmark.json")

    best_cmb = max(
        [("SFT", cmb_sft["accuracy"]), ("DPO", cmb_dpo["accuracy"]), ("MPO", cmb_mpo["accuracy"])],
        key=lambda x: x[1],
    )
    best_ceval = max([("SFT", ceval_sft["accuracy"]), ("DPO", ceval_dpo["accuracy"])], key=lambda x: x[1])
    best_red_coverage = max(
        [("SFT", red_sft["key_action_coverage"]), ("DPO", red_dpo["key_action_coverage"]), ("MPO", red_mpo["key_action_coverage"])],
        key=lambda x: x[1],
    )
    lowest_unsafe_aligned = min(
        [("SFT", red_sft["unsafe_action_rate"]), ("DPO", red_dpo["unsafe_action_rate"]), ("MPO", red_mpo["unsafe_action_rate"])],
        key=lambda x: x[1],
    )
    throughput = max(x["decode_tokens_per_second"] for x in kvcache["results"])

    lines = [
        "# Resume Claim Audit 2026-06-15",
        "",
        "## Verdict",
        "",
        "This project supports a resume claim about **SFT + preference alignment engineering**, not a claim that full reinforcement learning solved the task.",
        "DPO and MPO are alignment / preference-optimization experiments in this project. MPO is a custom mixed objective, not an on-policy RL algorithm such as PPO, GRPO, or GSPO.",
        "",
        "## Supported Results",
        "",
        "| Item | Base | Best aligned result | Delta | Supported wording |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| CMB-val accuracy | {pct(cmb_base['accuracy'])} | {best_cmb[0]} {pct(best_cmb[1])} | {pp(best_cmb[1], cmb_base['accuracy'])} | medical QA accuracy improved on CMB-val |",
        f"| C-Eval medical accuracy | {pct(ceval_base['accuracy'])} | {best_ceval[0]} {pct(best_ceval[1])} | {pp(best_ceval[1], ceval_base['accuracy'])} | medical exam subset accuracy improved |",
        f"| Red-flag key-action coverage | {pct(red_base['key_action_coverage'])} | {best_red_coverage[0]} {pct(best_red_coverage[1])} | {pp(best_red_coverage[1], red_base['key_action_coverage'])} | safety-action coverage improved |",
        f"| Red-flag JSON format rate | {pct(red_base['json_format_rate'])} | SFT/DPO/MPO {pct(1.0)} | {pp(1.0, red_base['json_format_rate'])} | structured safety response format improved |",
        f"| Red-flag unsafe action rate | {pct(red_base['unsafe_action_rate'])} | lowest aligned: {lowest_unsafe_aligned[0]} {pct(lowest_unsafe_aligned[1])} | {pp(lowest_unsafe_aligned[1], red_base['unsafe_action_rate'])} | report as residual safety risk, not as pure improvement |",
        f"| KV-cache benchmark | - | {throughput:.0f} tokens/s at best measured concurrency | - | deployment throughput benchmark completed |",
        "",
        "## Claims To Avoid",
        "",
        "- Avoid: \"reinforcement learning significantly improved medical safety\".",
        "- Avoid: \"MPO is a complete RLHF/GRPO/PPO implementation\".",
        "- Avoid: \"red-flag safety was fully solved\" because unsafe-action rate did not monotonically improve.",
        "- Avoid: public benchmark or clinical-safety claims beyond the listed internal evaluations.",
        "",
        "## Safer Resume Wording",
        "",
        "2026.02-2026.04 Qwen2.5-3B Chinese Medical QA and Red-Flag Safety Alignment",
        "",
        "- Built a Chinese medical LLM post-training pipeline on Qwen2.5-3B-Instruct with LLaMA-Factory, covering LoRA/QLoRA SFT, DPO preference alignment, a custom MPO mixed objective, automatic evaluation, LoRA merge, and deployment benchmarking.",
        "- Integrated CMB, Huatuo, DISC-Med, MedicalGPT, CMExam and red-flag triage data; constructed a million-scale candidate pool and ran controlled stage-1 experiments on 30k SFT samples and 25k preference pairs.",
        f"- Improved CMB-val accuracy from {pct(cmb_base['accuracy'])} to {pct(best_cmb[1])} and C-Eval medical subset accuracy from {pct(ceval_base['accuracy'])} to {pct(best_ceval[1])}; compared SFT, DPO, QLoRA and MPO ablations under the same evaluation scripts.",
        f"- Built red-flag safety evaluation for emergency escalation, key-action coverage, unsafe-action detection and JSON format stability; identified residual unsafe-action regressions and added a rule-based safety guardrail for deployment-time repair.",
        f"- Completed prediction generation, answer extraction, LoRA merge and KV-cache benchmarking; measured up to {throughput:.0f} tokens/s in the local concurrency-8 decode benchmark with about {kvcache['peak_torch_memory_gb']:.2f} GB peak memory.",
        "",
        "## Project Repair Notes",
        "",
        "- Keep DPO/MPO in the project, but describe them as preference alignment / mixed-objective optimization.",
        "- Treat red-flag SFT/DPO as high-coverage but not fully safe; use the safety guardrail and unsafe-rate metric as evidence of engineering maturity.",
        "- If a resume line mentions RL, add a new verified GRPO/KTO/ORPO experiment first; otherwise keep the wording as alignment rather than RL.",
    ]

    out = ROOT / "reports" / "resume_claim_audit_20260615.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
