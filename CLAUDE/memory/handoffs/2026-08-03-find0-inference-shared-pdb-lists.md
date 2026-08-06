# Handoff: Find_0 推理公共 PDB 清单简化

Date: 2026-08-03

## Current State

`训练与运行/sh/infer/` 的正式推理入口已经改为读取所有 Stage1 模型共用的三份 PDB 清单。修改位于 `Learn/CUMULATIVE` 工作区，未运行服务器脚本、未生成正式推理产物，也未由本侧修改 Git 索引。

公共清单计划写入：

- `/storage/penghongen/AdaLigand_stage1_inference/calibration_pdb_ids.json`
- `/storage/penghongen/AdaLigand_stage1_inference/validation_pdb_ids.json`
- `/storage/penghongen/AdaLigand_stage1_inference/train_pdb_ids.json`

Find_0 本次推理的人工记录计划写入 `/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/manifest.json`。该文件只记录路径、模型选择指标和启用参数，任何推理脚本都不读取它，也不把它作为运行条件。

## Completed

- `Find_0_prepare_inputs.sh` 已在工作区改名为 `prepare_inference_pdb_lists.sh`。
- 准备脚本只从已经过滤的 calibration、validation、train 集合提取 `pdb_id`，转为小写、去重、排序并写出三份公共 JSON 清单。
- 删除本轮 `infer/` 入口中的全部哈希计算、哈希记录、哈希校验、`final_keep_list.jsonl` 二次检查、拒绝覆盖和原子发布机制。
- calibration、validation、train 的 probability、F1 和 CLG 脚本均改为读取推理根目录下的公共清单。
- `manifest.json` 只保留 checkpoint、训练配置、数据目录、输出目录、公共清单路径、模型选择指标与启用参数。
- `训练与运行/sh/infer/README.md` 已同步公共清单位置、manifest 职责、执行顺序和新脚本名。
- `训练与运行/sh/infer/` 当前工作树不再包含旧的模型专属 `inputs/` 路径或任何哈希相关文字；全部 Shell 脚本通过 `bash -n`，`git diff --check` 通过。

## Decisions

- Pocket_Plus 项目不使用哈希校验、哈希身份或哈希运行门槛；来源可以用人能直接阅读和修改的路径记录。
- PDB 清单属于数据集合划分，不属于某个模型；所有 Stage1 模型共用同一套 calibration、validation、train 清单。
- 原始 split 文件可能为同一 PDB 保存多个 BOX 请求，因此不能直接复制给推理入口；只保留提取、规范化、去重和排序这一步必要转换。
- `manifest.json` 只是人工记录，不承担兼容、身份判断、覆盖保护或启动校验职责。

## Open Questions

- 当前 Git 索引由并行主任务保存了旧版 `Find_0_prepare_inputs.sh` 和旧版 probability 入口；其中仍有哈希逻辑和旧 `inputs/` 路径。符合本 handoff 的版本位于未暂存工作区，不能直接提交当前索引。

## Next Actions

1. 主任务收口时先核对 Git 索引，不要提交其中的旧准备脚本。
2. 以当前工作区的 `prepare_inference_pdb_lists.sh` 和全部已适配推理脚本为准，重新选择需要暂存的文件。
3. 正式运行前再次阅读 `训练与运行/sh/infer/README.md`；本 handoff 不授权提交服务器任务。

## Files To Reopen

- `训练与运行/sh/infer/prepare_inference_pdb_lists.sh`
- `训练与运行/sh/infer/README.md`
- `训练与运行/sh/infer/Find_0_calibration_probability.sh`
- `训练与运行/sh/infer/Find_0_freeze_thresholds.sh`
- `训练与运行/sh/infer/Find_0_calibration_F1.sh`
- `训练与运行/sh/infer/Find_0_validation_F1.sh`
- `训练与运行/sh/infer/Find_0_train_F1.sh`
