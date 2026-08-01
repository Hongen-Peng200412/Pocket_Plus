# Handoff: Stage1 centered 契约补齐与历史收口

Date: 2026-08-02

## Current State

Find centered 归档、Selector 消费、校准指标、测试和三份主要说明文档已经同步到新契约。用户审核实现和测试结果后，Pocket_Plus 的既有暂存修改与本轮修改已融入原提交职责；受影响的本地提交和分支按旧→新关系一对一重建，没有增加提交或分支，父节点数量与顺序保持不变。

## Completed

- centered Find 归档新增 `A_feat_L0`，保存原始 `float32 (L_A,49)` 输入特征，并按 `A_global_index` 对齐。
- Selector 直接读取 `A_feat_L0`，删除 `recover_a_feat_l0` 和 `receptor_tokens.npz/feat` 二次读取。
- 校准指标使用 `semantic_dice_micro_t_F1`，并新增按 PDB 等权平均的 `semantic_dice_macro_t_F1`。
- artifact、inference、Selector README、Find YAML 注释、学习注释和 AdaLigand 契约文档已经同步。
- centered 契约的直接相关测试为 `77 passed`；加入 `max_voxels=2046` 默认值与 component lineage 后，相关完整回归为 `87 passed`。
- 排除两个缺 `.project-root` 而无法收集的既有文件后，扩展测试 `374 passed, 3 failed`，三个失败均因当前缺少 CPC v3 配置文件。
- 历史重建候选覆盖 53 个受影响提交和 12 个本地分支；候选主端点树与审核通过的文件树相同，父节点数量、顺序和提交主题逐提交核验通过。

## Decisions

- 不修改 Stage1-Find 模型或配置中的现有前向结构。
- 正式推理尚未开始，因此不保留旧字段别名，不增加旧归档兼容或迁移分支。
- F1、CLG 与 Selected 的概率和特征均来自当前 centered 重跑；F1/CLG 的权威体素集合来自原滑窗森林，Selected 是按来源节点阈值重新细化的新集合。
- `candidate_voxel_index` 是归档条目内部的体素值局部索引；节点编号只在 `tree_id` 内唯一。
- `max_voxels` 使用全量 `Q95=682` 的 3.0 倍并向上取整，当前正式值为 `2046`；历史记录中的 `1023` 仅表示此前采用的 1.5 倍初始值。
- 历史重建已经完成；远端引用未修改，旧提交仍可能由远端引用或 Git 引用变更日志到达。

## Open Questions

- 是否另开任务补齐仓库级测试前置条件 `.project-root` 与 CPC v3 配置；它们不属于本轮范围。

## Next Actions

1. 正式 Stage1 推理开始前，使用当前 `max_voxels=2046` 运行 calibration 与组件生产。
2. 若要处理 `.project-root` 或 CPC v3 配置缺口，应另开独立任务，不与本轮契约修改混合。

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
