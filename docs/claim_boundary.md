# 结论边界

## 可以写进项目介绍的内容

- 基于 Qwen2.5-3B-Instruct 完成中文医疗问答后训练流程。
- 使用 LLaMA-Factory 跑通 LoRA/QLoRA SFT 和 DPO。
- 实现 MPO mixed objective 和 verifiable GRPO 对照实验。
- 构建 CMB/C-Eval 医学选择题自动评测流程。
- 在 CMB-test 3000 noleak 上，SFT 从 67.67% 提升到 72.27%。
- 在 CMB 8.2k/3k 严格题面去重的同分布诊断切分上，GRPO v12 从 72.27% 提升到 73.30%。
- 对红旗症状安全响应构建 emergency/routine mixed-risk 评测，发现并报告过度急诊化问题。
- 完成 LoRA 合并、KV Cache benchmark 和部署侧 guardrail 修补。

## 不建议写的内容

- 不写“达到临床安全”。
- 不写“官方 CMB leaderboard 刷榜成功”。
- 不写“DPO/MPO/GRPO 在所有任务上都显著优于 SFT”。
- 不把红旗症状 100% 急症召回解释成安全问题已解决，因为 routine 误报仍然明显。
- 不把 v12 的 CMB-test 内部切分写成官方 test 外部评测。

## 推荐简历表述

2026.02-2026.04 MedAlign-RLBench: Qwen2.5-3B 医疗问答与红旗症状安全响应对齐

- 基于 Qwen2.5-3B-Instruct 构建中文医疗问答后训练流程，覆盖数据清洗、LoRA/QLoRA SFT、DPO/MPO 对照、verifiable GRPO 诊断实验、自动评测、LoRA 合并与推理 benchmark。
- 在 CMB-test 3000 noleak 上，SFT 准确率由 67.67% 提升至 72.27%；进一步构造 CMB 8.2k/3k 严格题面去重诊断切分，GRPO 将 held-out 准确率提升至 73.30%。
- 针对红旗症状安全响应构建 emergency/routine 混合风险评测，统计急症召回、routine 特异性、过度急诊率、JSON 格式稳定性和 unsafe action，并加入部署侧 guardrail 修补。
- 完成 LoRA 合并与 KV Cache benchmark，在并发 8 推理测试下 decode 吞吐约 443 tokens/s，峰值显存约 5.87 GB。
