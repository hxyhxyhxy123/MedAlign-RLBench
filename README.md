# MedAlign-RLBench

基于 `Qwen2.5-3B-Instruct` 的中文医疗问答后训练项目。项目围绕医学选择题问答任务，完成数据处理、LoRA/QLoRA SFT、DPO/MPO 对照实验、verifiable GRPO 诊断实验、自动评测、LoRA 合并和 KV Cache 推理 benchmark。

本仓库是整理后的 GitHub 发布版，只保留代码、配置、结果摘要和复现实验入口。原始数据、完整训练集、模型权重、LoRA adapter、完整预测文件和运行日志不进入 Git。

## 技术栈

| 模块 | 内容 |
| --- | --- |
| 基座模型 | Qwen2.5-3B-Instruct |
| 训练框架 | PyTorch, Transformers, PEFT, TRL, LLaMA-Factory |
| 微调方法 | LoRA, QLoRA, SFT |
| 对齐方法 | DPO, IPO/ORPO 对照, MPO mixed objective, verifiable GRPO |
| 评测任务 | CMB-val, CMB-test 3000 noleak, C-Eval medical subset |
| 部署验证 | LoRA merge, `generate(use_cache=True)`, KV Cache throughput benchmark |

## 项目结构

```text
MedAlign-RLBench/
|-- configs/              # LLaMA-Factory、MPO、DeepSpeed、部署配置
|-- scripts/              # 数据构建、训练、评测、GRPO、benchmark 脚本
|-- experiments/          # SFT/DPO/MPO/GRPO 关键实验入口
|-- results/              # 可公开的指标摘要
|-- docs/                 # 实验说明、流程图、上传指南
|-- index.html            # GitHub Pages 项目主页
|-- README_EN.md          # English version
|-- Makefile              # 常用命令入口
```

## 主要结果

### 医学选择题评测

| 评测集 | Base | SFT | DPO | Best ablation | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| CMB-val | 57.14% | 65.71% | 65.36% | 65.71% | 280 题 |
| C-Eval 医疗子集 | 65.56% | 70.00% | 70.00% | - | 90 题 |
| CMB-test 3000 noleak | 67.67% | 72.27% | 72.27% | 72.23% | 固定评测切分 |

SFT 在医学选择题任务上带来稳定提升；DPO、MPO 和 hard-set preference 方法作为对照实验保留。

### 对齐实验对比

| 实验 | 训练设计 | 3000 noleak 准确率 | 相比 SFT |
| --- | --- | ---: | ---: |
| SFT baseline | stage-1 LoRA SFT | 72.27% | 0.00 |
| DPO final | preference pairs | 72.27% | 0.00 |
| MPO | mixed objective | 72.13% | -0.13 |
| Hard-DPO | SFT-wrong hard set | 72.10% | -0.17 |
| RS-DPO | rejection sampling preference | 72.23% | -0.03 |
| GRPO v10 | hard prompt GRPO | 71.00% | -1.27 |
| **GRPO v12** | disjoint held-out diagnostic split | **73.30%** | **+1.03** |

### GRPO v12 结果

| 模型 | Overall | Single | Multi | Invalid |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 72.27% | 75.87% | 41.59% | 3 |
| GRPO v12 ckpt-240 | 72.90% | 75.38% | 51.75% | 2 |
| GRPO v12 ckpt-360 | 72.93% | 75.42% | 51.75% | 2 |
| **GRPO v12 ckpt-480** | **73.30%** | **75.79%** | **52.06%** | 2 |
| GRPO v12 ckpt-600/final | 73.03% | 75.49% | 52.06% | 2 |

v12 使用可验证选择题 reward，最佳 checkpoint 为 `checkpoint-480`。这组实验用于分析偏好/RL 方法在同分布 held-out 场景下的增益，主要提升来自多选题。

### KV Cache 推理 benchmark

| 并发 | 输出 tokens | 耗时 | Decode throughput |
| --- | ---: | ---: | ---: |
| 1 | 256 | 3.99 s | 64.13 tokens/s |
| 4 | 1024 | 4.65 s | 220.12 tokens/s |
| 8 | 2048 | 4.62 s | 443.14 tokens/s |

峰值显存约 `5.87 GB`。

## 快速开始

无卡环境准备：

```bash
make bootstrap-nogpu
make inspect-data
make seed-data
make ready
```

有卡环境训练：

```bash
make gpu-setup
make sft-lora
make dpo
make mpo
```

GRPO 实验：

```bash
bash experiments/run_verifiable_grpo_v12.sh
```

评测与部署 benchmark：

```bash
make eval-choice
make bench-kvcache
```

## 文档

- `docs/architecture.md`：项目流程图
- `docs/experiment_notes.md`：实验设计说明
- `docs/github_upload_guide.md`：GitHub 上传与 Pages 配置
- `results/`：公开指标摘要

## English Version

See [README_EN.md](README_EN.md).

## Disclaimer

本项目仅用于机器学习工程实验与评测研究，不构成医疗产品，不能用于临床诊断或治疗决策。
