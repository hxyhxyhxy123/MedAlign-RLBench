# MedAlign-RLBench

A Qwen2.5-3B post-training project for Chinese medical QA and red-flag symptom safety response. The project covers data processing, LoRA/QLoRA SFT, DPO/MPO preference optimization, verifiable GRPO diagnostics, automatic evaluation, LoRA merging, and KV Cache inference benchmarking.

This repository is a cleaned public release. Raw datasets, processed training files, model weights, LoRA adapters, full prediction files, and runtime logs are excluded from Git.

## Stack

| Module | Details |
| --- | --- |
| Base model | Qwen2.5-3B-Instruct |
| Training | PyTorch, Transformers, PEFT, TRL, LLaMA-Factory |
| Fine-tuning | LoRA, QLoRA, SFT |
| Alignment | DPO, IPO/ORPO ablations, MPO mixed objective, verifiable GRPO |
| Evaluation | CMB-val, CMB-test noleak split, C-Eval medical subset, red-flag OOD |
| Deployment | LoRA merge, `generate(use_cache=True)`, KV Cache throughput benchmark |

## Main Results

| Evaluation | Base | SFT | DPO | Best ablation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CMB-val | 57.14% | 65.71% | 65.36% | 65.71% | 280 questions |
| C-Eval medical subset | 65.56% | 70.00% | 70.00% | - | 90 questions |
| CMB-test 3000 noleak | 67.67% | 72.27% | 72.27% | 72.23% | fixed internal evaluation |
| CMB-test full | - | 71.29% | - | - | 11200 questions, SFT only |

SFT gives the most stable gain, improving the 3000-row CMB noleak evaluation by `+4.60 pp` over the base model. Early DPO/MPO runs did not reliably outperform SFT on the external-style split, so they are reported as ablations rather than the main result.

## GRPO Diagnostic Experiment

| Model | Overall | Single | Multi | Invalid |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 72.27% | 75.87% | 41.59% | 3 |
| GRPO v12 ckpt-240 | 72.90% | 75.38% | 51.75% | 2 |
| GRPO v12 ckpt-360 | 72.93% | 75.42% | 51.75% | 2 |
| **GRPO v12 ckpt-480** | **73.30%** | **75.79%** | **52.06%** | 2 |
| GRPO v12 ckpt-600/final | 73.03% | 75.49% | 52.06% | 2 |

The v12 result is an in-distribution diagnostic split, not an official CMB leaderboard score. It uses a strict question-text-disjoint `8.2k/3k` split carved from CMB-test to verify whether GRPO can improve held-out accuracy when the training and evaluation distributions match. The main gain comes from multi-choice questions.

## Safety Evaluation

The red-flag safety evaluation includes emergency OOD cases, routine low-risk OOD cases, and mixed-risk evaluation. The results show improved emergency recall but also reveal over-triage on routine cases.

| Model | Risk accuracy | Emergency recall | Routine specificity | Routine over-triage |
| --- | ---: | ---: | ---: | ---: |
| Base | 63.19% | 26.39% | 100.00% | 0.00% |
| SFT | 50.00% | 100.00% | 0.00% | 100.00% |
| DPO | 50.00% | 100.00% | 0.00% | 100.00% |
| MPO | 66.67% | 100.00% | 33.33% | 66.67% |

## Usage

No-GPU preparation:

```bash
make bootstrap-nogpu
make inspect-data
make seed-data
make ready
```

Training:

```bash
make gpu-setup
make sft-lora
make dpo
make mpo
```

GRPO diagnostic run:

```bash
bash experiments/run_verifiable_grpo_v12.sh
```

Evaluation and deployment benchmark:

```bash
make eval-choice
make build-redflag-ood
make eval-redflag-ood-strict
make guard-redflag-ood
make bench-kvcache
```

## Claim Boundary

Supported claims:

- Qwen2.5-3B LoRA/QLoRA SFT, DPO/MPO ablations, and verifiable GRPO diagnostics were implemented.
- SFT improves medical choice QA accuracy on CMB/C-Eval style evaluations.
- In-distribution GRPO v12 improves a strict held-out diagnostic split from 72.27% to 73.30%.
- Red-flag evaluation covers emergency recall, routine specificity, over-triage, JSON stability, and unsafe-action checks.
- LoRA merge and KV Cache inference benchmarking were completed.

Not claimed:

- Clinical safety.
- Official CMB leaderboard performance.
- DPO/MPO consistently outperform SFT on all external evaluations.
- Red-flag safety is fully solved.

## Disclaimer

This project is for machine learning engineering experiments and evaluation research only. It is not a medical product and must not be used for clinical diagnosis or treatment decisions.
