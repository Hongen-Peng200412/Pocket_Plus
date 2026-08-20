# Stage1 V3 推理输入与产物概览

本文从正式命令输入开始, 按数据进入代码的顺序说明 Stage1 V3 推理会读取和写出什么。字段级当前权威最终迁入 `src/inference/README.md` 与 AdaLigand `文档/规划文档/BOX-level数据契约.md`; 本文负责冷读导航。

形状记号：`N_entry` 是一个 centered 文件的候选数，`L_voxel` 是全部候选来源体素拼接后的长度，`L_A` 与 `L_P` 分别是 Find A 原子表与 P 点表拼接后的长度。

## 输入

| 输入 | 一个输入对应的实体 | 推理用途 |
| --- | --- | --- |
| Hydra resolved config 与 checkpoint | 一个 Stage1 训练运行 | 恢复 `unet_c1` 或一个 Find wrapper |
| `<density_root>/<pdb_id>/exp.npy` 与 `exp.npz` | 一个 PDB 的实验密度完整图和几何 | 构造所有 producer 的 80³ 输入, 保存 F3 的 `experimental_density_48` |
| `<density_root>/<pdb_id>/sim.npy` 与 `sim.npz` | 一个 PDB 的受体模拟密度完整图和几何 | 构造 Find 输入, 保存所有 F3 的 `simulated_density_48` |
| `<dataset_root>/parse/<pdb_id>/receptor_tokens.npz` | 一个 PDB 的完整受体原子表 | 构造 Find 的 A 输入和 50 维 `A_feat_L0` |
| V3 split 清单 | 一个数据划分, 例如 `calibration` | 决定 PDB 顺序 |
| occurrence 配体区域资产 | 一个 PDB 的真实配体实例掩码 | calibration 与评估, 不进入模型输入 |

`Stage1Dataset` 从完整图 mmap 中裁出 80³ BOX。Find 的 `atom_feat (N_A,49)` 与 `atom_is_backbone (N_A,)` 在模型输入边界拼成 50 维; `receptor_tokens.npz` 本身不改写。

## 完整图概率

`probability/probability_map.npz` 对应一个 producer、一个数据划分和一个 PDB:

| 字段 | 数据类型与形状 | 含义 |
| --- | --- | --- |
| `probability_map` | `float32`, `(D,H,W)` | 完整 ZYX 网格上的滑窗 Gaussian 融合配体概率, 不乘受体 hardmask |
| `origin_xyz` | `float32`, `(3,)` | 完整图 voxel-grid corner 的世界 XYZ 原点, 单位 Å |
| `voxel_size_xyz` | `float32`, `(3,)` | 世界 XYZ 三轴体素尺寸, 单位 Å/voxel |

科学 NPZ 只有以上三个字段。`probability/geometry.json` 保存 `full_shape_zyx`、`origin_xyz`、`voxel_size_xyz`、`window_shape_zyx`、显式 `stride_zyx`、`gaussian_sigma` 和 `window_count`；`status/probability/performance.json` 单独保存计时。

## 连通区域

`blobs/F1_blobs.npz` 与 `blobs/F3_blobs.npz` 结构相同。前者使用语义 micro-F1 阈值, 后者使用语义 micro-F3 阈值。

| 字段 | 数据类型与形状 | 含义 |
| --- | --- | --- |
| `blob_index` | `int32`, `(N_blob,)` | 当前文件按稳定排序得到的预测连通区域编号 |
| `voxel_offsets` | `int64`, `(N_blob+1,)` | 同时切分 `voxel_index_global_zyx` 与 `source_probability`; 首值为 0, 末值为总区域体素数 |
| `voxel_index_global_zyx` | `int32`, `(L_voxel,3)` | 全部区域拼接后的完整图离散 ZYX 坐标 |
| `source_probability` | `float32`, `(L_voxel,)` | 与全图坐标逐体素对齐的完整图融合概率 |
| `source_probability_mean` | `float32`, `(N_blob,)` | 每个区域的 `source_probability` 算术平均 |
| `voxel_count` | `int32`, `(N_blob,)` | 每个区域的体素数, 等于相邻 `voxel_offsets` 之差 |
| `fits_centered_box` | `bool`, `(N_blob,)` | True 表示存在一个无 padding 80³ BOX 可以容纳区域全部体素 |
| `centered_box_start_zyx` | `int32`, `(N_blob,3)` | `fits_centered_box=true` 时的完整图 BOX 起点; False 时使用 `-1` |
| `source_threshold_value` | `float32`, `(1,)` | 当前文件使用的 F1 或 F3 体素阈值 |

所有阈值连通区域都保留。`min_voxels` 不改变本文件。

## centered 共同字段

`centered/F1_basic.npz` 和 `centered/F3_centered.npz` 对应一个 PDB 的一组候选。候选轴长度为 `N_entry`。

| 字段 | 数据类型与形状 | 含义 |
| --- | --- | --- |
| `source_blob_index` | `int32`, `(N_entry,)` | 指向同 PDB 对应 blobs 文件的 `blob_index` |
| `centered_box_index` | `int32`, `(N_entry,)` | 当前 centered 文件中从 0 开始连续的条目编号 |
| `box_start_zyx` | `int32`, `(N_entry,3)` | 80³ BOX 在完整图中的离散 ZYX 起点 |
| `box_shape_zyx` | `uint8`, `(N_entry,3)` | 固定为 `(80,80,80)` |
| `box_origin_world` | `float32`, `(N_entry,3)` | 80³ BOX corner 的世界 XYZ 坐标, 单位 Å |
| `voxel_size_world` | `float32`, `(N_entry,3)` | 世界 XYZ 体素尺寸, 单位 Å/voxel |
| `source_probability_mean` | `float32`, `(N_entry,)` | 来源 blobs 文件的区域平均概率 |
| `source_threshold_value` | `float32`, `(N_entry,)` | 来源 blobs 文件使用的语义 F1 或 F3 阈值 |
| `score` | `float32`, `(N_entry,)` | 当前冻结评分参数得到的候选分数 |
| `selected` | `bool`, `(N_entry,)` | 当前候选是否同时达到冻结分数阈值和 `min_voxels` |
| `voxel_offsets` | `int64`, `(N_entry+1,)` | 以半开区间切分 `voxel_index_local_zyx`、`source_probability`、`centered_probability` 和 F3 可选 `voxel_final`; 首值 0, 末值 L_voxel |
| `voxel_index_local_zyx` | `int16`, `(L_voxel,3)` | 来源区域体素在当前 80³ BOX 内的离散 ZYX 坐标 |
| `source_probability` | `float32`, `(L_voxel,)` | 同一来源区域的完整图融合概率 |
| `centered_probability` | `float32`, `(L_voxel,)` | 当前 80³ BOX 重新前向后在权威体素处取出的概率 |

F1 basic 的正式文件到此结束，不保存辅助受体头、`voxel_final`、48³ 数组或 A/P 表。

F3 另外保存 `voxel_aux_offsets`、`voxel_aux_index_local_zyx` 和 `voxel_aux_probability`。`voxel_aux_offsets` 以半开区间切分后两个数组, 首值为 0, 末值为 L_aux。`voxel_final` 为 `float16 (L_voxel,C_voxel)`，与 `voxel_index_local_zyx`、`source_probability` 和 `centered_probability` 逐行对齐。

## F3 的 48³ 字段

正式 F3 对每个候选保存:

| 字段 | 数据类型与形状 | 含义 |
| --- | --- | --- |
| `v_centroid_local_zyx` | `float32`, `(N_entry,3)` | 完整权威 V 整数下标在 80³ 局部 ZYX 坐标中的算术平均 |
| `crop_start_local_zyx` | `int16`, `(N_entry,3)` | V-centered 48³ 在 80³ BOX 内的实际 ZYX 起点, 每轴位于 `[0,32]` |
| `crop_center_offset_zyx` | `float32`, `(N_entry,3)` | V 来源体素质心减去实际 48³ 几何中心的 ZYX 偏移, 单位 voxel |
| `crop_clipped_axis_mask` | `bool`, `(N_entry,3)` | True 表示该轴的请求起点被边界限制 |
| `experimental_density_48` | `float32`, `(N_entry,48,48,48)` | 当前 48³ 范围的原始实验密度 |
| `simulated_density_48` | `float32`, `(N_entry,48,48,48)` | 同一范围的原始受体模拟密度 |
| `source_probability_48` | `float32`, `(N_entry,48,48,48)` | 同一范围的完整图融合概率 |

## Find F3 的 A/P 字段

Find 的 F3 保存 `A_offsets` 与 `P_offsets`。`A_offsets` 以半开区间切分 `A_global_index`、`A_coord_local_xyz`、`A_coord_centered_world`、`A_probability`、`A_feat_L0`、`A_feat_L1`、`A_feat_L2` 和 `A_feat_L3`，首值为 0，末值为 L_A。`P_offsets` 以半开区间切分 `P_coord_local_xyz`、`P_probability`、`P_feat_L2` 和 `P_feat_L3`，首值为 0，末值为 L_P。

- A 表保存 `A_global_index`、`A_coord_local_xyz`、`A_coord_centered_world`、`A_probability`、`A_feat_L0`、`A_feat_L1`、`A_feat_L2`、`A_feat_L3`。`A_feat_L0` 为 `float32 (L_A,50)`; 最后一维是主链标志。
- P 表保存 `P_coord_local_xyz`、`P_probability`、`P_feat_L2`、`P_feat_L3`。
- A/P 学习特征使用 float16, 概率与坐标使用 float32。`unet_c1` 和任何 F1 基本版都不出现 A/P 字段。

## 校准与评估产物

`calibration/semantic_threshold_scan.npz` 保存完整阈值编号、TP、FP、FN 和 F1/F3 曲线。`calibration/stage1_v3.json` 保存 checkpoint 规范化绝对路径、两类语义阈值、F1/F3 选择参数和三阶段搜索的获胜事实；`stage1_v3.metrics.json` 保存两个角色的最终指标。全部载荷完成后最后发布 `calibration/_COMPLETE`。

每个 PDB 的 `evaluation/<role>.npz` 保存阈值轴、top-K 轴、按分数排序的全部 centered 候选、`candidate_selected`、候选与 occurrence 的交集矩阵、两侧体素数、逐候选语义 TP、双向覆盖命中掩码、各阈值的完整候选轴下标/`occurrence_id` 轴下标，以及 top-K 获胜事实。`topk_winning_candidate_rank` 是已选候选序列中从 0 开始的名次，不是完整候选轴下标；`topk_winning_occurrence_index` 是 occurrence 轴下标。数据划分级 `<role>.jsonl` 保存逐 PDB 指标，`<role>.metrics.json` 保存 micro 与 PDB 等权 macro 汇总。
