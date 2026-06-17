# MedAlign-RLBench

面向中文医疗问答和红旗症状安全响应的 Qwen2.5-3B 后训练实验项目。项目以 `Qwen2.5-3B-Instruct` 为基座，围绕 CMB 医学选择题和红旗症状分诊场景，搭建了数据构建、LoRA/QLoRA SFT、DPO/MPO 偏好优化、verifiable GRPO 诊断实验、自动评测、LoRA 合并和 KV Cache 推理 benchmark 的完整流程。

本仓库是整理后的 GitHub 发布版，只保留代码、配置、结果摘要和实验报告。原始数据、完整训练集、模型权重、LoRA adapter、完整预测文件和运行日志不进入 Git。

## 技术栈

| 模块 | 使用内容 |
| --- | --- |
| 基座模型 | Qwen2.5-3B-Instruct |
| 训练框架 | PyTorch, Transformers, PEFT, TRL, LLaMA-Factory |
| 微调方法 | LoRA, QLoRA, SFT |
| 对齐方法 | DPO, IPO/ORPO 对照, MPO mixed objective, verifiable GRPO |
| 评测任务 | CMB-val, CMB-test noleak split, C-Eval medical subset, red-flag OOD |
| 部署验证 | LoRA merge, `generate(use_cache=True)`, KV Cache throughput benchmark |

## 项目结构

```text
MedAlign-RLBench/
|-- configs/              # LLaMA-Factory、MPO、DeepSpeed、部署配置
|-- scripts/              # 数据构建、训练、评测、GRPO、guardrail、benchmark 脚本
|-- experiments/          # 关键实验入口脚本，包含 DPO/MPO/GRPO 对照
|-- results/              # 可公开的指标摘要，不包含完整预测文件
|-- docs/                 # 实验说明、结论边界、上传指南、面试问答
|-- index.html            # GitHub Pages 项目主页
|-- README.md             # 中文说明
|-- README_EN.md          # English version
|-- Makefile              # 常用命令入口
```

## 核心结果

### 1. 标准医学选择题评测

| 评测集 | Base | SFT | DPO | MPO/Best ablation | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| CMB-val | 57.14% | 65.71% | 65.36% | 65.71% | 280 题 |
| C-Eval 医疗子集 | 65.56% | 70.00% | 70.00% | - | 90 题 |
| CMB-test 3000 noleak | 67.67% | 72.27% | 72.27% | 72.23% | 3000 条内部固定评测 |
| CMB-test full | - | 71.29% | - | - | 11200 题，仅 SFT 跑完整评测 |

结论：SFT 是最稳定的主线收益，在 CMB-test 3000 noleak 上相对 Base 提升 `+4.60 pp`。早期 DPO/MPO 在外部分布评测上没有稳定超过 SFT，因此没有把它包装成主结果。

### 2. 偏好优化和 GRPO 对照

| 实验 | 训练数据/设计 | 3000 noleak 准确率 | 相比 SFT | 结论 |
| --- | --- | ---: | ---: | --- |
| SFT baseline | 30k stage-1 SFT | 72.27% | 0.00 | 主基线 |
| DPO final | preference pairs | 72.27% | 0.00 | 没有超过 SFT |
| MPO | mixed objective | 72.13% | -0.13 | 作为消融保留 |
| Hard-DPO | SFT wrong hard set | 72.10% | -0.17 | 没有转化为准确率收益 |
| RS-DPO | rejection sampling preference | 72.23% | -0.03 | 接近 SFT |
| GRPO v10 | CMExam-style hard prompts | 71.00% | -1.27 | reward 可优化但不迁移 |
| GRPO v12 diagnostic | CMB 8.2k/3k disjoint split | **73.30%** | **+1.03** | 同分布诊断实验成功 |

v12 的最佳 checkpoint 为 `checkpoint-480`。它不是官方 CMB-test leaderboard 结果，而是从 CMB-test 构造的严格题面去重 `8.2k/3k` 同分布诊断切分，用来验证“RL 不涨点是否主要来自训练/评测分布不一致”。

### 3. GRPO v12 诊断结果

| 模型 | Overall | Single | Multi | Invalid |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 72.27% | 75.87% | 41.59% | 3 |
| GRPO v12 ckpt-240 | 72.90% | 75.38% | 51.75% | 2 |
| GRPO v12 ckpt-360 | 72.93% | 75.42% | 51.75% | 2 |
| **GRPO v12 ckpt-480** | **73.30%** | **75.79%** | **52.06%** | 2 |
| GRPO v12 ckpt-600/final | 73.03% | 75.49% | 52.06% | 2 |

这组实验说明：当训练数据和评测数据保持同分布，并使用可验证选择题 reward 时，GRPO 可以带来可见提升；其中主要收益来自多选题。

### 4. 红旗症状安全响应

项目构建了急症 OOD、routine 低风险 OOD 和混合风险评测。早期只测急症样本时结果过高，加入 routine 样本后发现模型存在“过度急诊化”问题，因此报告中保留了该限制。

| 模型 | 风险判断准确率 | 急症召回 | routine 特异性 | routine 过度急诊率 |
| --- | ---: | ---: | ---: | ---: |
| Base | 63.19% | 26.39% | 100.00% | 0.00% |
| SFT | 50.00% | 100.00% | 0.00% | 100.00% |
| DPO | 50.00% | 100.00% | 0.00% | 100.00% |
| MPO | 66.67% | 100.00% | 33.33% | 66.67% |

结论：安全对齐能提高急症召回和格式稳定性，但没有完全解决低风险 routine 场景下的误报问题。仓库保留 `redflag_safety_guard.py` 作为部署侧 guardrail 修补。

### 5. KV Cache 推理 benchmark

| 并发 | 输出 tokens | 耗时 | Decode throughput |
| --- | ---: | ---: | ---: |
| 1 | 256 | 3.99 s | 64.13 tokens/s |
| 4 | 1024 | 4.65 s | 220.12 tokens/s |
| 8 | 2048 | 4.62 s | 443.14 tokens/s |

峰值显存约 `5.87 GB`。该 benchmark 用于说明部署侧 `use_cache=True` 下的解码吞吐，不作为服务端压测结论。

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

GRPO 诊断实验：

```bash
bash experiments/run_verifiable_grpo_v12.sh
```

评测与部署 benchmark：

```bash
make eval-choice
make build-redflag-ood
make eval-redflag-ood-strict
make guard-redflag-ood
make bench-kvcache
```

## 结论边界

可以支持的说法：

- 完成了 Qwen2.5-3B 医疗问答场景的 LoRA/QLoRA SFT、DPO/MPO 对照和 verifiable GRPO 诊断实验。
- SFT 在 CMB/C-Eval 医学选择题上带来稳定提升。
- DPO/MPO/GRPO 被用于对照和问题定位；v12 同分布 GRPO 在 3000 条 held-out 诊断集上超过 SFT。
- 安全响应评测包含急症召回、routine 特异性、过度急诊率、JSON 格式稳定性和 unsafe action 检查。
- 完成 LoRA 合并和 KV Cache 推理 benchmark。

不建议声称：

- 项目达到临床安全标准。
- v12 结果是官方 CMB leaderboard 成绩。
- DPO/MPO 在所有外部评测上显著超过 SFT。
- 红旗症状安全问题已经被完全解决。

## English Version

See [README_EN.md](README_EN.md).

## Disclaimer

本项目仅用于机器学习工程实验与评测研究，不构成医疗产品，不能用于临床诊断或治疗决策。
