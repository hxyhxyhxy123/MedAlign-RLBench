# Configs

配置文件分为四类：

| 目录/文件 | 说明 |
| --- | --- |
| `llamafactory/` | SFT、QLoRA、DPO、IPO/ORPO、red-flag 等 LLaMA-Factory 训练配置 |
| `deepspeed/zero2.json` | DeepSpeed ZeRO-2 配置 |
| `mpo_recipe.yaml` | 自定义 MPO mixed objective 配置 |
| `deploy_kvcache.yaml` | LoRA merge 和 KV Cache benchmark 配置 |
| `data_manifest.yaml` | 数据源和本地路径清单 |

公开仓库不包含真实数据文件。复现前需要先下载/转换数据，并检查配置里的本地路径。
