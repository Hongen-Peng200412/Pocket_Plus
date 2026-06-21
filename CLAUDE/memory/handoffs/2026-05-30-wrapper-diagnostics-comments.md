# Handoff: Wrapper Diagnostics 与注释收尾

Date: 2026-05-30

## Current State

Stage1 wrapper 相关文件仍处于本地修改状态。本轮主要在已有修复基础上完成两件事：修正 diagnostics 中 adaptive `p_sampling` 的阈值统计语义，并按用户要求补充 wrapper pipeline 中关键张量/统计的简短 inline comments。

## Completed

- `src/wrappers/voxel_point_stage1_diagnostics.py`：修正 diagnostics 里 adaptive `p_sampling` 统计逻辑，使阈值比较使用实际用于采样分支判定的 `sampling_target`，避免在存在 top-k / gating 语义时诊断统计和真实采样依据错位。
- `src/wrappers/voxel_point_stage1.py`：补充关键变量注释，覆盖 `source_folder_idx`、`atom_mask`、`receptor_mask`、`ligand_target`、`ligand_valid`、`candidate_outputs`、`dense_num_gt`。
- 保持注释短而贴近张量语义，没有扩展到大段说明或重构。
- 已运行语法验证：`python -m py_compile "src/wrappers/voxel_point_stage1.py" "src/wrappers/voxel_point_stage1_diagnostics.py"`，通过且无输出。

## Decisions

- comments 只补在后续阅读最容易误解的 wrapper pipeline 张量和诊断统计上，不把整段流程改成叙述式注释。
- diagnostics 中保留 `sampling_target` 作为关键中间语义，便于区分真实采样依据和只用于日志/显示的概率张量。

## Open Questions

- 尚未运行完整训练 smoke；此前记录的 `dataset/fused` Hydra 配置缺失问题仍可能阻塞相关实验启动。

## Next Actions

- 如继续推进 Stage1 wrapper，本地下一步可以跑更接近训练路径的 focused smoke，前提是先决定恢复 `dataset/fused` 配置，或把实验指向当前存在的数据集配置。
- 若用户只关心本轮注释/diagnostics 修复，当前可进入 diff review 或提交准备。

## Files To Reopen

- `src/wrappers/voxel_point_stage1.py`
- `src/wrappers/voxel_point_stage1_diagnostics.py`
- `CLAUDE/memory/projects/pocket-plus.json`
