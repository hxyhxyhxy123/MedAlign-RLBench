# MedAlign-RLBench

A Qwen2.5-3B post-training project for Chinese medical QA. The project covers data processing, LoRA/QLoRA SFT, DPO/MPO ablations, verifiable GRPO diagnostics, automatic evaluation, LoRA merging, and KV Cache inference benchmarking.

This repository is a cleaned public release. Raw datasets, processed training files, model weights, LoRA adapters, full prediction files, and runtime logs are excluded from Git.

## Stack

| Module | Details |
| --- | --- |
| Base model | Qwen2.5-3B-Instruct |
| Training | PyTorch, Transformers, PEFT, TRL, LLaMA-Factory |
| Fine-tuning | LoRA, QLoRA, SFT |
| Alignment | DPO, IPO/ORPO ablations, MPO mixed objective, verifiable GRPO |
| Evaluation | CMB-val, CMB-test 3000 noleak, C-Eval medical subset |
| Deployment | LoRA merge, `generate(use_cache=True)`, KV Cache throughput benchmark |

## Main Results

| Evaluation | Base | SFT | DPO | Best ablation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CMB-val | 57.14% | 65.71% | 65.36% | 65.71% | 280 questions |
| C-Eval medical subset | 65.56% | 70.00% | 70.00% | - | 90 questions |
| CMB-test 3000 noleak | 67.67% | 72.27% | 72.27% | 72.23% | fixed evaluation split |

SFT gives a stable improvement on medical choice QA. DPO, MPO, and hard-set preference methods are kept as controlled ablations.

## Alignment Ablations

| Experiment | Training design | 3000 noleak accuracy | Delta vs SFT |
| --- | --- | ---: | ---: |
| SFT baseline | stage-1 LoRA SFT | 72.27% | 0.00 |
| DPO final | preference pairs | 72.27% | 0.00 |
| MPO | mixed objective | 72.13% | -0.13 |
| Hard-DPO | SFT-wrong hard set | 72.10% | -0.17 |
| RS-DPO | rejection sampling preference | 72.23% | -0.03 |
| GRPO v10 | hard prompt GRPO | 71.00% | -1.27 |
| **GRPO v12** | disjoint held-out diagnostic split | **73.30%** | **+1.03** |

## GRPO v12

| Model | Overall | Single | Multi | Invalid |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 72.27% | 75.87% | 41.59% | 3 |
| GRPO v12 ckpt-240 | 72.90% | 75.38% | 51.75% | 2 |
| GRPO v12 ckpt-360 | 72.93% | 75.42% | 51.75% | 2 |
| **GRPO v12 ckpt-480** | **73.30%** | **75.79%** | **52.06%** | 2 |
| GRPO v12 ckpt-600/final | 73.03% | 75.49% | 52.06% | 2 |

v12 uses a verifiable choice reward. The best checkpoint is `checkpoint-480`, with most of the gain coming from multi-choice questions.

## KV Cache Benchmark

| Concurrency | Output tokens | Time | Decode throughput |
| --- | ---: | ---: | ---: |
| 1 | 256 | 3.99 s | 64.13 tokens/s |
| 4 | 1024 | 4.65 s | 220.12 tokens/s |
| 8 | 2048 | 4.62 s | 443.14 tokens/s |

Peak GPU memory is about `5.87 GB`.

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

GRPO run:

```bash
bash experiments/run_verifiable_grpo_v12.sh
```

Evaluation and benchmark:

```bash
make eval-choice
make bench-kvcache
```

## Disclaimer

This project is for machine learning engineering experiments and evaluation research only. It is not a medical product and must not be used for clinical diagnosis or treatment decisions.
