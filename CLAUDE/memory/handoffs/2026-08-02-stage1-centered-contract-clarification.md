# Stage1 centered 契约补齐与审核交接

## Current State

Find centered 归档、Selector 消费、校准指标、测试和三份主要说明文档已经同步到新契约。动手前已有的暂存修改仍保留在暂存区；本轮任务修改保留在未暂存区。用户明确要求先审核实现和测试结果，因此尚未重构任何 Git 提交。

## Completed

- centered Find 归档新增 `A_feat_L0`，保存原始 `float32 (L_A,49)` 输入特征，并按 `A_global_index` 对齐。
- Selector 直接读取 `A_feat_L0`，删除 `recover_a_feat_l0` 和 `receptor_tokens.npz/feat` 二次读取。
- 校准指标使用 `semantic_dice_micro_t_F1`，并新增按 PDB 等权平均的 `semantic_dice_macro_t_F1`。
- artifact、inference、Selector README、Find YAML 注释、学习注释和 AdaLigand 契约文档已经同步。
- 本轮直接相关测试 `77 passed`；排除两个缺 `.project-root` 而无法收集的既有文件后，扩展测试 `374 passed, 3 failed`，三个失败均因当前缺少 CPC v3 配置文件。

## Decisions

- 不修改 Stage1-Find 模型或配置中的现有前向结构。
- 正式推理尚未开始，因此不保留旧字段别名，不增加旧归档兼容或迁移分支。
- F1、CLG 与 Selected 的概率和特征均来自当前 centered 重跑；F1/CLG 的权威体素集合来自原滑窗森林，Selected 是按来源节点阈值重新细化的新集合。
- `candidate_voxel_index` 是归档条目内部的体素值局部索引；节点编号只在 `tree_id` 内唯一。
- `max_voxels` 使用全量 `Q95=682` 的 3.0 倍并向上取整，当前正式值为 `2046`；历史记录中的 `1023` 仅表示此前采用的 1.5 倍初始值。
- 用户已审核实现并授权一对一重构提交；重构不得新增最终提交或分支，也不得改变原有拓扑。

## Open Questions

- 是否另开任务补齐仓库级测试前置条件 `.project-root` 与 CPC v3 配置；它们不属于本轮范围。

## Next Actions

1. 执行 Pocket_Plus 历史重建并核验实现端点、学习端点和原有拓扑。
2. 重建完成后再次运行相关测试与暂存区/树对象核验。

## Files To Reopen

- `src/artifacts/io.py`
- `src/inference/centered.py`
- `src/evaluation/calibration.py`
- `src/selector/dataset.py`
- `src/artifacts/readme.md`
- `src/inference/README.md`
- `src/selector/README.md`
- `tests/evaluation/test_stage1_calibration.py`
- `tests/inference/test_stage1_centered.py`
- `tests/selector/test_dataset_and_freeze.py`
- `../AdaLigand/文档/讨论/BOX-level数据契约.md`
