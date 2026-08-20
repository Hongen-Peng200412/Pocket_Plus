# Stage1 V3 推理模块

本目录是从 Stage1 训练 checkpoint 到完整图概率、`F1_basic.npz`、`F3_centered.npz` 和实例评估的唯一活动主线。旧 Selector、组件森林、CLG、Li 和七种 Fα 实现不保留兼容入口；需要考察时使用 Git 历史。

## 阅读顺序与职责

| 顺序 | 文件 | 主要职责 |
| --- | --- | --- |
| 1 | `artifacts.py` | 固定路径、无 object dtype NPZ、JSON/JSONL 原子发布 |
| 2 | `checkpoint.py` | 从训练 run 的代码快照和 resolved config 恢复 wrapper |
| 3 | `full_map.py` | 80³ 滑窗、GPU 前向、异步 D2H 和确定性 Gaussian 融合 |
| 4 | `blobs.py` | F1/F3 单阈值 26 邻域连通区域 |
| 5 | `centered.py` | 候选 BOX 重新前向、V/A/P 稀疏表和 48³ 稠密数组 |
| 6 | `scoring.py` | 来源平均概率和 Find A 原子 Gaussian 分数 |
| 7 | `evaluation.py` | 逐候选交集事实、语义、双向覆盖、一对一和 top-K 指标 |
| 8 | `calibration.py` | 语义直方图与 centered 选择参数冻结 |
| 9 | `pipeline.py` | 单 PDB 概率、centered 和评分发布事务 |
| 10 | `workflow.py` | 跨 PDB 有界流水、calibration 和冻结参数运行 |
| 11 | `cli.py` | `calibrate`/`run` 子命令与运行身份构造 |

`cli.py` 只解析参数和构造 Dataset/wrapper。`workflow.py` 直接编排生产阶段，不建立 runner 工厂或 producer 回调层。

## 五类 PDB 科学产物

`probability/probability_map.npz` 的字段精确为：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `probability_map` | `float32 (D,H,W)` | 不乘受体 hardmask 的有限配体概率 |
| `origin_xyz` | `float32 (3,)` | 完整网格角点的世界 XYZ 坐标，单位 Å |
| `voxel_size_xyz` | `float32 (3,)` | XYZ 体素尺寸，单位 Å/voxel |

窗口形状、显式 `stride_zyx`、规范化 Gaussian sigma 和窗口数写入 `probability/geometry.json`；计时只写 `status/probability/performance.json`，不混入科学 NPZ。

`F1_blobs.npz` 与 `F3_blobs.npz` 保存阈值下的全部 26 邻域连通区域，精确字段为：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `blob_index` | `int32 (N_blob,)` | 稳定排序后的连续身份 |
| `voxel_offsets` | `int64 (N_blob+1,)` | 切分两个逐体素值表 |
| `voxel_index_global_zyx` | `int32 (L_voxel,3)` | 完整图 ZYX 体素索引 |
| `source_probability` | `float32 (L_voxel,)` | 与体素索引逐项对齐的完整图概率 |
| `source_probability_mean` | `float32 (N_blob,)` | 每个 blob 的正式平均概率和第一排序键 |
| `voxel_count` | `int32 (N_blob,)` | 每个 blob 的体素数 |
| `fits_centered_box` | `bool (N_blob,)` | 包围盒能否由合法 80³ BOX 容纳 |
| `centered_box_start_zyx` | `int32 (N_blob,3)` | 合法完整图 ZYX 起点；不可容纳时为 `-1` |
| `source_threshold_value` | `float32 (1,)` | 本角色语义阈值 |

连通区域阶段不应用 `min_voxels`。

`F1_basic.npz` 的精确字段为：

- 候选级：`centered_box_index int32 (N,)`、`source_blob_index int32 (N,)`、`box_start_zyx int32 (N,3)`、`box_shape_zyx uint8 (N,3)`、`box_origin_world float32 (N,3)`、`voxel_size_world float32 (N,3)`、`source_probability_mean float32 (N,)`、`source_threshold_value float32 (N,)`、`score float32 (N,)`、`selected bool (N,)`；
- 稀疏体素：`voxel_offsets int64 (N+1,)` 切分 `voxel_index_local_zyx int16 (L,3)`、`source_probability float32 (L,)` 和 `centered_probability float32 (L,)`。

它显式关闭 `voxel_final`、A/P、auxiliary 受体概率和三张 48³ 数组。

`F3_centered.npz` 包含上述全部共同字段，并增加以下精确字段：

- V 与 auxiliary：`voxel_final float16 (L_voxel,C_voxel)`；`voxel_aux_offsets int64 (N+1,)` 切分 `voxel_aux_index_local_zyx int16 (L_aux,3)` 和 `voxel_aux_probability float32 (L_aux,)`；
- 48³：`v_centroid_local_zyx float32 (N,3)`、`crop_start_local_zyx int16 (N,3)`、`crop_center_offset_zyx float32 (N,3)`、`crop_clipped_axis_mask bool (N,3)`，以及 `experimental_density_48`、`simulated_density_48`、`source_probability_48` 三个 `float32 (N,48,48,48)` 数组；
- Find A 表：`A_offsets int64 (N+1,)` 切分 `A_global_index int64 (N_A,)`、`A_coord_local_xyz float32 (N_A,3)`、`A_coord_centered_world float32 (N_A,3)`、`A_probability float32 (N_A,)`、`A_feat_L0 float32 (N_A,50)` 和 `A_feat_L1/L2/L3 float16 (N_A,C_A*)`；
- Find P 表：`P_offsets int64 (N+1,)` 切分 `P_coord_local_xyz float32 (N_P,3)`、`P_probability float32 (N_P,)` 和 `P_feat_L2/L3 float16 (N_P,C_P*)`。

`A_feat_L0` 是 float32 `(N_A,50)`：前 49 维来自 `receptor_tokens.npz:feat`，最后一维来自同一原子的 `is_backbone`。A 表只保留核心 80³ 内且到来源 blob 最近体素中心不超过 10 Å 的原子；Gaussian 评分再使用固定 5 Å 截断。学习得到的 V/A/P 特征落盘为 float16，原始密度和 `A_feat_L0` 保持 float32。

## 校准与指标

语义阈值扫描把概率量化为 `floor(p * denominator)`，分别冻结 calibration 全集的 micro-F1 和 micro-F3 首个最大值。完整 TP、FP、FN 和 F-beta 曲线写入 `calibration/semantic_threshold_scan.npz`。

基本模式与 U-Net 完整模式先扫描实际出现的来源平均概率阈值，再冻结阈值并扫描 `min_voxels=8..40`。Find 完整模式按固定顺序执行：

1. 500 组 `tau_angstrom × lambda_positive × lambda_negative × gauss_score_min` 粗网格；
2. 固定首轮 tau，围绕两个 lambda 的 5×5 乘数和分数下限的 15 个乘数形成 375 组细网格；
3. 冻结 Gaussian 参数，只扫描 `min_voxels=8..40`。

F1 basic 最大化 semantic、coverage@0.3 和 one-to-one@0.3 三个 micro-F1 之和；F3 centered 最大化对应三个 micro-F2 之和。评估另外报告覆盖阈值 0.3/0.5/0.6、top-3/4/5、micro 和 PDB 等权 macro。逐 PDB NPZ保存交集矩阵、覆盖命中掩码、每个阈值的一对一匹配行列和 top-K 获胜候选/occurrence 身份。

`calibration/stage1_v3.json` 绑定 producer、checkpoint SHA-256、resolved config SHA-256、推理配置 SHA-256、模型代码来源和语义分母。模型代码来源必须由命令显式选择 `current_workspace` 或 `training_snapshot`。快照模式先在 run 的 `src_snapshot/src` 中完成模型与 wrapper 导入, 再恢复当前工作区路径并导入 V3 Dataset。`run` 必须逐项匹配该身份；全部校准文件发布后才建立 `calibration/_COMPLETE`。

## 并行与发布

每个 PDB 内部：CPU 线程提前物化 batch，唯一主线程拥有 GPU，D2H 结果由单独 CPU 线程按提交顺序融合或整理。

跨 PDB：一个完整图离开 GPU 后，概率 NPZ 压缩和 F1/F3 blobs 在 CPU 执行，GPU 立即开始下一 PDB；centered NPZ 压缩同样与下一 PDB 的 centered GPU 前向重叠。`pending_probability_pdbs` 与 `pending_centered_pdbs` 限制尚未发布的大数组数量，避免 train 清单造成内存累积。

推理物化线程共享同一个 Dataset 和 mmap LRU。缓存命中后复用同一 PDB 的完整图；首次并发 miss 允许多个线程分别打开同一 mmap。`_ByteLruCache` 只在 `get()`/`put()` 的短 OrderedDict 临界区持有可重入锁，保证 LRU 顺序和字节计数一致；实际 NPY 裁块和密度通道计算不在锁内。

正式 NPZ、JSON 和 JSONL 都在最终目录建立临时文件并通过 `os.replace` 发布。大型 NPZ 不做重复解压重读。F3 候选数严格大于显式 `blob_limit` 时只写 `_BLOB_EXCEED` 事实并继续生产，不产生跳过终态。浮点字段只在正式发布边界检查 NaN/Inf；代码不建立额外 `_valid*` 防御层。
