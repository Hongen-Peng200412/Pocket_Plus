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
| 11 | `cli.py` | `calibrate`/`run` 子命令、Dataset/wrapper 构造与 checkpoint 路径传递 |

`cli.py` 只解析参数和构造 Dataset/wrapper。`workflow.py` 直接编排生产阶段，不建立 runner 工厂或 producer 回调层。

## 五类 PDB 科学产物

形状记号：`D/H/W` 是完整图 Z/Y/X 轴长；`N_blob` 是一个 blobs 文件的区域数；`N` 是一个 centered 文件的候选数；`L_voxel`/`L_aux` 是 offsets 拼接后的来源/辅助体素数；`N_A`/`N_P` 是 Find A 原子/P 点总数；`C_voxel` 与 `C_A*`/`C_P*` 是对应模型层的特征宽度。

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
| `voxel_offsets` | `int64 (N_blob+1,)` | 以半开区间切分 `voxel_index_global_zyx` 与 `source_probability`；首值 0，末值 L_voxel |
| `voxel_index_global_zyx` | `int32 (L_voxel,3)` | 完整图 ZYX 体素索引 |
| `source_probability` | `float32 (L_voxel,)` | 与体素索引逐项对齐的完整图概率 |
| `source_probability_mean` | `float32 (N_blob,)` | 每个 blob 的正式平均概率和第一排序键 |
| `voxel_count` | `int32 (N_blob,)` | 每个 blob 的体素数 |
| `fits_centered_box` | `bool (N_blob,)` | 包围盒能否由合法 80³ BOX 容纳 |
| `centered_box_start_zyx` | `int32 (N_blob,3)` | 合法完整图 ZYX 起点；不可容纳时为 `-1` |
| `source_threshold_value` | `float32 (1,)` | 本角色语义阈值 |

连通区域阶段不应用 `min_voxels`。

`F1_basic.npz` 的精确字段为：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `centered_box_index` | `int32 (N,)` | 当前文件内连续 centered 编号 |
| `source_blob_index` | `int32 (N,)` | 对应 blobs 文件中的 `blob_index` |
| `box_start_zyx` | `int32 (N,3)` | 80³ BOX 在完整图中的 ZYX 起点 |
| `box_shape_zyx` | `uint8 (N,3)` | 固定为 `(80,80,80)` |
| `box_origin_world` | `float32 (N,3)` | BOX 角点世界 XYZ 坐标，单位 Å |
| `voxel_size_world` | `float32 (N,3)` | 世界 XYZ 体素尺寸，单位 Å/voxel |
| `source_probability_mean` | `float32 (N,)` | 来源 blob 的完整图平均概率 |
| `source_threshold_value` | `float32 (N,)` | 来源 blob 使用的语义阈值 |
| `score` | `float32 (N,)` | calibration 冻结定义得到的候选分数 |
| `selected` | `bool (N,)` | 是否同时达到冻结分数和最小体素数 |
| `voxel_offsets` | `int64 (N+1,)` | 以半开区间切分 `voxel_index_local_zyx`、`source_probability` 和 `centered_probability`；首值 0，末值 L_voxel |
| `voxel_index_local_zyx` | `int16 (L_voxel,3)` | 来源 blob 体素在 80³ BOX 内的 ZYX 索引 |
| `source_probability` | `float32 (L_voxel,)` | 与体素索引对齐的完整图概率 |
| `centered_probability` | `float32 (L_voxel,)` | 同一体素的 centered 重算概率 |

它显式关闭 `voxel_final`、A/P、auxiliary 受体概率和三张 48³ 数组。

`F3_centered.npz` 包含上述全部共同字段，并增加以下字段：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `voxel_final` | `float16 (L_voxel,C_voxel)` | 来源体素的 V 学习特征 |
| `voxel_aux_offsets` | `int64 (N+1,)` | 以半开区间切分 `voxel_aux_index_local_zyx` 和 `voxel_aux_probability`；首值 0，末值 L_aux |
| `voxel_aux_index_local_zyx` | `int16 (L_aux,3)` | auxiliary 受体体素的 BOX-local ZYX 索引 |
| `voxel_aux_probability` | `float32 (L_aux,)` | 独立辅助受体概率，不改变配体概率 |
| `v_centroid_local_zyx` | `float32 (N,3)` | 来源 blob 的局部整数 ZYX 体素下标算术平均，单位 voxel |
| `crop_start_local_zyx` | `int16 (N,3)` | 48³ 裁块在 80³ BOX 内的 ZYX 起点 |
| `crop_center_offset_zyx` | `float32 (N,3)` | 来源 blob 质心相对 48³ 裁块中心的 ZYX 偏移，单位 voxel |
| `crop_clipped_axis_mask` | `bool (N,3)` | 48³ 起点是否在对应轴受 80³ 边界限制 |
| `experimental_density_48` | `float32 (N,48,48,48)` | 实验密度裁块，后三轴按 ZYX 排列 |
| `simulated_density_48` | `float32 (N,48,48,48)` | 模拟密度裁块，后三轴按 ZYX 排列 |
| `source_probability_48` | `float32 (N,48,48,48)` | 完整图概率裁块，后三轴按 ZYX 排列 |
| `A_offsets` | `int64 (N+1,)` | 以半开区间切分 `A_global_index`、`A_coord_local_xyz`、`A_coord_centered_world`、`A_probability`、`A_feat_L0`、`A_feat_L1`、`A_feat_L2` 和 `A_feat_L3`；首值 0，末值 N_A；仅 Find 存在 |
| `A_global_index` | `int64 (N_A,)` | `receptor_tokens.npz` 第一维的全局原子编号 |
| `A_coord_local_xyz` | `float32 (N_A,3)` | A 原子的 BOX-local XYZ 体素坐标 |
| `A_coord_centered_world` | `float32 (N_A,3)` | A 原子相对 80³ BOX 世界中心的 XYZ 位移，单位 Å |
| `A_probability` | `float32 (N_A,)` | A 原子配体概率 |
| `A_feat_L0` | `float32 (N_A,50)` | 49 维 token 与 `is_backbone` 拼接结果 |
| `A_feat_L1` | `float16 (N_A,C_A1)` | 第一层 A 学习特征 |
| `A_feat_L2` | `float16 (N_A,C_A2)` | 第二层 A 学习特征 |
| `A_feat_L3` | `float16 (N_A,C_A3)` | 第三层 A 学习特征 |
| `P_offsets` | `int64 (N+1,)` | 以半开区间切分 `P_coord_local_xyz`、`P_probability`、`P_feat_L2` 和 `P_feat_L3`；首值 0，末值 N_P；仅 Find 存在 |
| `P_coord_local_xyz` | `float32 (N_P,3)` | P 点的 BOX-local XYZ 体素坐标 |
| `P_probability` | `float32 (N_P,)` | P 点配体概率 |
| `P_feat_L2` | `float16 (N_P,C_P2)` | 第二层 P 学习特征 |
| `P_feat_L3` | `float16 (N_P,C_P3)` | 第三层 P 学习特征 |

`A_feat_L0` 是 float32 `(N_A,50)`：前 49 维来自 `receptor_tokens.npz:feat`，最后一维来自同一原子的 `is_backbone`。A 表只保留核心 80³ 内且到来源 blob 最近体素中心不超过 10 Å 的原子；Gaussian 评分再使用固定 5 Å 截断。学习得到的 V/A/P 特征落盘为 float16，原始密度和 `A_feat_L0` 保持 float32。

## 校准与指标

语义阈值扫描把概率量化为 `floor(p * denominator)`，分别冻结 calibration 全集的 micro-F1 和 micro-F3 首个最大值。`calibration/semantic_threshold_scan.npz` 的字段为：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `denominator` | `int32` 标量 | 概率阈值网格分母 |
| `beta_values` | `float64 (N_beta,)` | F-beta 的 beta 轴，当前依次为 1 和 3 |
| `threshold_grid_index` | `int32 (denominator+1,)` | 从 0 到 denominator 的阈值整数编号 |
| `f_beta_curve` | `float64 (N_beta,denominator+1)` | beta 轴与阈值轴组成的完整 micro F-beta 曲线 |
| `tp` | `int64 (denominator+1,)` | 每个包含端点阈值的跨 PDB TP |
| `fp` | `int64 (denominator+1,)` | 每个包含端点阈值的跨 PDB FP |
| `fn` | `int64 (denominator+1,)` | 每个包含端点阈值的跨 PDB FN |

`calibration/stage1_v3.json` 的嵌套字段为：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `checkpoint_path` | 字符串 | 当前 calibration 使用的 checkpoint 规范化绝对路径 |
| `semantic.denominator` | 整数 | 与扫描 NPZ 的 `denominator` 相同 |
| `semantic.positive_voxel_count` | 整数 | calibration 全集真实配体体素数 |
| `semantic.negative_voxel_count` | 整数 | calibration 全集真实背景体素数 |
| `semantic.thresholds.<F-beta>.grid_index` | 整数 | 首个达到最大 micro F-beta 的阈值整数编号 |
| `semantic.thresholds.<F-beta>.value` | 浮点数 | `grid_index / denominator` 得到的概率阈值 |
| `semantic.thresholds.<F-beta>.micro_f_beta` | 浮点数 | 获胜阈值在 calibration 全集上的 micro F-beta |
| `semantic.thresholds.<F-beta>.tp` | 整数 | 获胜阈值的跨 PDB 体素 TP |
| `semantic.thresholds.<F-beta>.fp` | 整数 | 获胜阈值的跨 PDB 体素 FP |
| `semantic.thresholds.<F-beta>.fn` | 整数 | 获胜阈值的跨 PDB 体素 FN |
| `roles.<role>.source_threshold` | 浮点数 | 当前角色 blobs 使用的完整图概率阈值 |
| `roles.<role>.selection.objective` | 浮点数 | 最终最小体素数对应的三项 micro F-beta 之和 |
| `roles.<role>.selection.objective_beta` | 浮点数 | F1 basic 为 1，F3 centered 为 2 |
| `roles.<role>.selection.score_mode` | 字符串 | `source_mean` 或 `find_gaussian` |
| `roles.<role>.selection.score_parameters.tau_angstrom` | 浮点数 | Find Gaussian 距离标准差；来源均值模式无此字段 |
| `roles.<role>.selection.score_parameters.lambda_positive` | 浮点数 | Find Gaussian 正项系数；来源均值模式无此字段 |
| `roles.<role>.selection.score_parameters.lambda_negative` | 浮点数 | Find Gaussian 负项系数；来源均值模式无此字段 |
| `roles.<role>.selection.score_threshold` | 浮点数 | 冻结候选分数下限，包含端点 |
| `roles.<role>.selection.min_voxels` | 整数 | 冻结来源 blob 最小体素数，包含端点 |
| `roles.<role>.selection.stages.score_threshold.objective` | 浮点数 | 来源均值实际分数扫描的最优目标值 |
| `roles.<role>.selection.stages.score_threshold.score_threshold` | 浮点数 | 来源均值实际分数扫描的最优阈值 |
| `roles.<role>.selection.stages.coarse.objective` | 浮点数 | Find Gaussian 粗网格的最优目标值 |
| `roles.<role>.selection.stages.coarse.tau_angstrom` | 浮点数 | Find Gaussian 粗网格的最优距离标准差 |
| `roles.<role>.selection.stages.coarse.lambda_positive` | 浮点数 | Find Gaussian 粗网格的最优正项系数 |
| `roles.<role>.selection.stages.coarse.lambda_negative` | 浮点数 | Find Gaussian 粗网格的最优负项系数 |
| `roles.<role>.selection.stages.coarse.score_threshold` | 浮点数 | Find Gaussian 粗网格的最优分数阈值 |
| `roles.<role>.selection.stages.refined.objective` | 浮点数 | Find Gaussian 细网格的最优目标值 |
| `roles.<role>.selection.stages.refined.tau_angstrom` | 浮点数 | Find Gaussian 细网格固定的距离标准差 |
| `roles.<role>.selection.stages.refined.lambda_positive` | 浮点数 | Find Gaussian 细网格的最优正项系数 |
| `roles.<role>.selection.stages.refined.lambda_negative` | 浮点数 | Find Gaussian 细网格的最优负项系数 |
| `roles.<role>.selection.stages.refined.score_threshold` | 浮点数 | Find Gaussian 细网格的最优分数阈值 |
| `roles.<role>.selection.stages.min_voxels.objective` | 浮点数 | 最终最小体素数对应的三项指标目标值 |
| `roles.<role>.selection.stages.min_voxels.min_voxels` | 整数 | 两种评分模式最终冻结的最小体素数 |

每个 `semantic.thresholds.<role>` 中，`grid_index` 是阈值整数编号，`value=grid_index/denominator`，`micro_f_beta` 是获胜指标，`tp/fp/fn` 是该阈值的跨 PDB 体素计数。每个 `selection` 中，`objective` 是三项 micro F-beta 之和，`objective_beta` 是 1 或 2，`score_mode` 是 `source_mean` 或 `find_gaussian`，`score_threshold` 与 `min_voxels` 都包含端点，`stages` 保存实际执行的阈值、粗网格、细网格或最小体素数阶段。Find 的 `score_parameters` 精确包含 `tau_angstrom`、`lambda_positive` 和 `lambda_negative`；来源均值模式使用空对象。

`calibration/_COMPLETE` 精确包含字符串 `checkpoint_path` 与固定字符串 `result_scope="calibration_fitted"`。`calibration/stage1_v3.metrics.json` 包含同一 `checkpoint_path` 和 `roles` 对象，`roles.F1_basic`、`roles.F3_centered` 分别保存下述跨 PDB指标。

基本模式与 U-Net 完整模式先扫描实际出现的来源平均概率阈值，再冻结阈值并扫描 `min_voxels=8..40`。Find 完整模式按固定顺序执行：

1. 500 组 `tau_angstrom × lambda_positive × lambda_negative × gauss_score_min` 粗网格；
2. 固定首轮 tau，围绕两个 lambda 的 5×5 乘数和分数下限的 15 个乘数形成 375 组细网格；
3. 冻结 Gaussian 参数，只扫描 `min_voxels=8..40`。

F1 basic 最大化 semantic、coverage@0.3 和 one-to-one@0.3 三个 micro-F1 之和；F3 centered 最大化对应三个 micro-F2 之和。评估另外报告覆盖阈值 0.3/0.5/0.6、top-3/4/5、micro 和 PDB 等权 macro。

每个 PDB 的 `evaluation/<role>.npz` 字段为：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `coverage_thresholds` | `float32 (N_threshold,)` | 双向覆盖阈值轴 |
| `topk_values` | `int32 (N_topk,)` | top-K 候选数量轴 |
| `occurrence_id` | `int32 (N_gt,)` | 真实 ligand occurrence 标识轴 |
| `source_blob_index` | `int32 (N_pred,)` | 按分数稳定降序的来源 blob 编号 |
| `candidate_score` | `float32 (N_pred,)` | 与候选轴对齐的冻结分数 |
| `candidate_selected` | `bool (N_pred,)` | 与候选轴对齐的最终选择掩码 |
| `intersections` | `int64 (N_pred,N_gt)` | 每对候选与 occurrence 的体素交集数 |
| `pred_sizes` | `int64 (N_pred,)` | 每个候选的体素数 |
| `gt_sizes` | `int64 (N_gt,)` | 每个 occurrence 的体素数 |
| `candidate_semantic_tp` | `int64 (N_pred,)` | 每个候选与真实 occurrence 并集的交集体素数 |
| `semantic_tp` | `int64` 标量 | 已选候选体素并集与真实体素并集的交集数 |
| `semantic_fp` | `int64` 标量 | 已选候选体素并集落在真实体素并集外的体素数 |
| `semantic_fn` | `int64` 标量 | 真实体素并集未被已选候选体素并集覆盖的体素数 |
| `coverage_pred_hit_mask` | `bool (N_threshold,N_pred)` | 每个完整候选是否命中至少一个 occurrence，与 selected 无关 |
| `coverage_gt_hit_mask` | `bool (N_threshold,N_gt)` | 每个 occurrence 是否被至少一个已选候选命中 |
| `one_to_one_match_offsets` | `int64 (N_threshold+1,)` | 按阈值切分 `one_to_one_match_pred_index` 与 `one_to_one_match_gt_index`；首值 0，末值 L_match |
| `one_to_one_match_pred_index` | `int32 (L_match,)` | 每个匹配在分数排序后完整候选轴上的下标 |
| `one_to_one_match_gt_index` | `int32 (L_match,)` | 与前项对齐的 occurrence 轴下标 |
| `topk_winning_candidate_rank` | `int32 (N_topk,N_threshold)` | 已选候选序列中从 0 开始的首个获胜名次；未命中为 -1 |
| `topk_winning_occurrence_index` | `int32 (N_topk,N_threshold)` | 与获胜名次对齐的 occurrence 轴下标；未命中为 -1 |

数据划分级 `evaluation/<role>.jsonl` 每个 PDB 保存一条 `pdb_id` 加指标映射；`evaluation/<role>.metrics.json` 保存同一指标映射的跨 PDB 版本。指标字段为：

| 字段格式 | 类型 | 含义 |
| --- | --- | --- |
| `pdb_count` | 整数 | 当前汇总中的 PDB 数量 |
| `semantic_tp` | 整数 | 全部 PDB 的语义 TP |
| `semantic_fp` | 整数 | 全部 PDB 的语义 FP |
| `semantic_fn` | 整数 | 全部 PDB 的语义 FN |
| `semantic_micro_f1` | 浮点数 | 先汇总全部 PDB 计数再计算的语义 F1 |
| `semantic_micro_f2` | 浮点数 | 先汇总全部 PDB 计数再计算的语义 F2 |
| `semantic_macro_f1` | 浮点数 | 逐 PDB 语义 F1 的算术平均 |
| `semantic_macro_f2` | 浮点数 | 逐 PDB 语义 F2 的算术平均 |
| `topk_eligible_pdb_count` | 整数 | 至少含一个真实 occurrence 的 PDB 数量 |
| `coverage_micro_precision_<t>` | 浮点数 | 覆盖阈值 t 下的跨 PDB 候选侧命中比例 |
| `coverage_micro_recall_<t>` | 浮点数 | 覆盖阈值 t 下的跨 PDB occurrence 侧命中比例 |
| `coverage_micro_f<beta>_<t>` | 浮点数 | 覆盖阈值 t 下的跨 PDB F1 或 F2 |
| `coverage_macro_f<beta>_<t>` | 浮点数 | 覆盖阈值 t 下的逐 PDB 等权 F1 或 F2 |
| `one_to_one_micro_precision_<t>` | 浮点数 | 覆盖阈值 t 下的一对一匹配 precision |
| `one_to_one_micro_recall_<t>` | 浮点数 | 覆盖阈值 t 下的一对一匹配 recall |
| `one_to_one_micro_f<beta>_<t>` | 浮点数 | 覆盖阈值 t 下的一对一跨 PDB F1 或 F2 |
| `one_to_one_macro_f<beta>_<t>` | 浮点数 | 覆盖阈值 t 下的一对一逐 PDB 等权 F1 或 F2 |
| `top<K>_success_count_<t>` | 整数 | 前 K 个已选候选至少命中一个 occurrence 的 PDB 数 |
| `top<K>_success_ratio_<t>` | 浮点数 | 成功 PDB 数除以 `topk_eligible_pdb_count` |

动态字段中的 `<beta>` 取 1 或 2。阈值 `<t>` 把小数点改成 `p`，例如 0.3 写成 `0p3`。

`calibration/stage1_v3.json` 保存当前 checkpoint 的规范化绝对路径、语义阈值和 F1/F3 选择参数。`run` 读取 `--calibration` 显式指定的 JSON 和同目录完成标记，只比较 checkpoint 路径与结果范围；calibration 目录可以不同于当前 `output_root`。代码不计算 checkpoint、配置或代码摘要。同一 checkpoint 采用不同 F3/F2 目标、阈值范围或其他科学参数时，命令用不同的 `output_root` 版本目录区分本次产物；目录名由使用者决定，代码不推断版本。模型代码来源仍由命令显式选择 `current_workspace` 或 `training_snapshot`：快照模式先从训练 run 的 `src_snapshot/src` 恢复模型与 wrapper，再恢复当前工作区路径并导入 V3 Dataset。全部校准文件发布后才建立 `calibration/_COMPLETE`。

## 并行与发布

每个 PDB 内部：CPU 线程提前物化 batch，唯一主线程拥有 GPU，D2H 结果由单独 CPU 线程按提交顺序融合或整理。

跨 PDB：一个完整图离开 GPU 后，概率 NPZ 压缩和 F1/F3 blobs 在 CPU 执行，GPU 立即开始下一 PDB；centered NPZ 压缩同样与下一 PDB 的 centered GPU 前向重叠。`pending_probability_pdbs` 与 `pending_centered_pdbs` 限制尚未发布的大数组数量，避免 train 清单造成内存累积。

推理物化线程共享同一个 Dataset 和 mmap LRU。缓存命中后复用同一 PDB 的完整图；首次并发 miss 允许多个线程分别打开同一 mmap。`_ByteLruCache` 只在 `get()`/`put()` 的短 OrderedDict 临界区持有可重入锁，保证 LRU 顺序和字节计数一致；实际 NPY 裁块和密度通道计算不在锁内。

正式 NPZ、JSON 和 JSONL 都在最终目录建立临时文件并通过 `os.replace` 发布。大型 NPZ 不做重复解压重读。F3 候选数严格大于显式 `blob_limit` 时只写 `_BLOB_EXCEED` 事实并继续生产，不产生跳过终态。浮点字段只在正式发布边界检查 NaN/Inf；代码不建立额外 `_valid*` 防御层。
