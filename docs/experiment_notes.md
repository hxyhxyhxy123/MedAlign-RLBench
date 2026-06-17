# 实验说明

## 主线流程

项目主线为：

```text
Qwen2.5-3B-Instruct -> LoRA/QLoRA SFT -> DPO/MPO/GRPO 对照 -> 自动评测 -> 部署 benchmark
```

主要实验包括：

1. 构建医学问答 stage-1 SFT 数据和偏好数据。
2. 使用 LLaMA-Factory 完成 LoRA/QLoRA SFT 与 DPO。
3. 实现 MPO mixed objective，作为偏好优化对照。
4. 构建 hard-set preference 和 rejection sampling preference 数据。
5. 使用 verifiable GRPO 优化选择题可验证 reward。
6. 使用统一脚本生成预测并计算 CMB/C-Eval 选择题指标。
7. 完成 LoRA 合并和 KV Cache 推理 benchmark。

## GRPO v12

v12 使用可验证选择题 reward，预测答案与标准答案按选项集合计算得分。最佳 checkpoint-480 在 3000 条 held-out 评测切分上达到 73.30%，高于 SFT baseline 的 72.27%。

该实验的主要收益来自多选题，说明可验证 reward 对多选集合匹配有较好的优化效果。
