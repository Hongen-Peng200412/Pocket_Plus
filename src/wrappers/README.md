# Stage1 训练与推理 wrapper

本目录把 Stage1 网络输出转换为训练损失、指标、日志和推理所需的稳定接口。正式入口是 `voxel_point_stage1.py`；Stage1 V3 推理只调用其中的 `forward_voxel_probability()` 或完整 `forward()`，不会复制训练模型。

## 阅读顺序

| 顺序 | 文件 | 职责 |
| --- | --- | --- |
| 1 | `voxel_point_stage1.py` | 构造网络、执行 voxel/point 前向、协调全局或 Find_1 双组梯度裁剪，并暴露训练与推理共同接口 |
| 2 | `voxel_point_stage1_losses.py` | 计算配体、主链和距离损失 |
| 3 | `voxel_point_stage1_metrics.py` | 计算训练与 validation 指标 |
| 4 | `voxel_point_stage1_logging.py` | 组织周期级日志字段 |
| 5 | `voxel_point_stage1_diagnostics.py` | 生成诊断统计 |
| 6 | `voxel_point_stage1_scheduler.py` | 按 validation 指标更新优化器调度状态 |
| 7 | `voxel_point_stage1_old.py` | 仅供 Git 内历史兼容阅读，不是 V3 推理入口 |

## Stage1 V3 推理接口

- `forward_voxel_probability(batch)` 返回未乘受体 hardmask 的配体 voxel logits，形状为 `float tensor (B,1,80,80,80)`。
- `forward(batch)` 的 `voxel_logits_ligand` 与前述 logits 同义；完整模式另外提供 `voxel_logits_aux`、`voxel_final`、A/P 坐标、概率 logits 和学习特征。
- `atom_global_indices` 在 `stage1_dataset.py` 的受体原子顺序中保持稳定；`A_feat_L0` 由 49 维受体 token 与 `is_backbone` 拼成 50 维。
- wrapper 不负责滑窗、连通区域、centered BOX 选择或 NPZ 发布；这些职责位于 `src/inference/`。
