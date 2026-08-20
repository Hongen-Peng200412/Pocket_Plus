# Stage1 V3 推理模块

本目录实现从训练 checkpoint 到 `F1_basic.npz`、`F3_centered.npz` 和实例评估的唯一活动主线。旧 Selector、CLG、组件森林、Li 和多套 Fα 实现不在本目录保留；Git 历史承担旧实现查阅职责。

## 阅读顺序

1. `artifacts.py`：先理解路径、原子发布和完成标记。
2. `checkpoint.py`：理解训练代码快照与当前工作区代码的显式选择。
3. `full_map.py`：理解 80³ 滑窗、Gaussian 融合和 GPU/CPU 重叠。
4. `blobs.py`：理解 F1/F3 单阈值 26-连通区域。
5. `centered.py`：理解候选重新前向、稀疏 V/A/P 表和 48³ 稠密裁块。
6. `scoring.py`：理解来源平均概率与 Find A 原子 Gaussian 分数。
7. `evaluation.py` 和 `calibration.py`：理解语义阈值、选择参数和 micro/macro 指标。
8. `pipeline.py` 和 `cli.py`：最后阅读产物装配和命令流程。

## 模块职责

| 文件 | 主要职责 | 正式入口 |
| --- | --- | --- |
| `artifacts.py` | 路径和原子发布 | `Stage1ArtifactPaths`、`load_stage1_npz`、`publish_stage1_artifact` |
| `checkpoint.py` | 恢复模型与最终训练配置 | `load_stage1_wrapper` |
| `full_map.py` | 完整图概率 | `infer_full_map` |
| `blobs.py` | 单阈值连通区域 | `extract_probability_blobs`、`publish_probability_blobs` |
| `centered.py` | F1/F3 centered 前向和归档 | `infer_centered_boxes`、`pack_centered_entries` |
| `scoring.py` | 候选评分和选择 | `score_centered_candidates`、`select_centered_candidates` |
| `evaluation.py` | 稀疏交集事实与指标 | `evaluate_centered_pdb`、`aggregate_stage1_metrics` |
| `calibration.py` | 阈值与选择参数冻结 | `calibrate_semantic_thresholds`、`tune_centered_selection` |
| `pipeline.py` | 单 PDB 产物阶段 | `produce_probability_map`、`produce_centered_role`、`score_and_publish_centered` |
| `cli.py` | calibration 与 run 命令 | `main` |

## 科学字段

完整图 `probability_map.npz` 保存 float32 `probability_map (D,H,W)`、完整图 ZYX 形状、世界 XYZ 原点和体素尺寸，以及显式滑窗步幅、规范化 Gaussian sigma 和窗口数量。概率不乘受体 hardmask。

`F1_blobs.npz` 与 `F3_blobs.npz` 保存阈值下的全部 26-连通区域，不按体素数删除：

- `blob_index (N_blob,)`；
- `voxel_offsets (N_blob+1,)`；
- `voxel_index_global_zyx (L_voxel,3)`；
- `source_probability (L_voxel,)`；
- `source_probability_mean (N_blob,)`；
- `voxel_count (N_blob,)`；
- `fits_centered_box (N_blob,)`；
- `centered_box_start_zyx (N_blob,3)`；
- `source_threshold_value (1,)`。

`F1_basic.npz` 保存来源 blob identity、80³ 几何、来源/重算概率、分数和选择标志；配置显式关闭 `voxel_final`、A/P 和 48³ 数组。

`F3_centered.npz` 在共同字段上增加：

- `voxel_final (L_voxel,C_voxel)`，float16；
- auxiliary receptor 稀疏坐标和概率；
- `experimental_density_48`、`simulated_density_48`、`source_probability_48`，float32 `(N_entry,48,48,48)`；
- Find producer 的 `A_offsets`、50 维 float32 `A_feat_L0`、A L1–L3、A 概率和坐标；
- Find producer 的 `P_offsets`、P L2–L3、P 概率和坐标。

`A_feat_L0` 前 49 维来自 Dataset 的 `atom_feat`，最后一维来自同一原子行的 `atom_is_backbone`。A 表只保留 80³ 核心内且到来源 blob 最近体素中心不超过 10 Å 的受体原子。Gaussian 评分再使用固定 5 Å 截断。

## 坐标和指标

完整图与 BOX 的离散数组都使用 ZYX 轴序；原子和世界坐标使用 XYZ 轴序。世界原点表示体素网格角点，体素中心在离散坐标上增加 `0.5`。

V-centered 48³ 起点是 `rint(mean(V_zyx)+0.5-24)`，逐轴裁剪到 `[0,32]`。三张 48³ 数组使用同一起点。

实例交集同时计算双向 coverage：`intersection / prediction_size` 与 `intersection / occurrence_size` 都达到阈值才算有效边。coverage 允许多对多；one-to-one 先按连续分数 `sqrt(c_pred*c_gt)` 做一次 Hungarian 配对，再对不同阈值计数。空分母指标固定为 `0.0`。

## 并行和发布

完整图阶段重叠 CPU 请求物化、页锁定 H2D、GPU 前向、异步 D2H 和有序 CPU Gaussian 融合。centered 阶段重叠下一批请求物化、当前 GPU 前向和上一批 CPU 归档整理。单进程只有主线程访问 GPU。

所有正式文件在最终目录中创建临时文件，重读后通过 `os.replace` 发布。`_COMPLETE` 只在最终 NPZ 已经替换后建立。
