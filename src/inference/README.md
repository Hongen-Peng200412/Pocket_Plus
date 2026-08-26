# Stage1 V3 推理模块

本目录是从 Stage1 训练 checkpoint 到完整图概率、任意 F-alpha blobs、任意 F-alpha centered 和实例评估的唯一活动主线。正式入口被拆成五个阶段，不保留旧 `calibrate/run`、固定 `F1_basic/F3_centered` 或旧 Selector、组件森林、CLG、Li 兼容入口。

## 代码组织与阅读顺序

| 顺序 | 文件 | 主要职责与入口 |
| --- | --- | --- |
| 1 | `artifacts.py` | `f_alpha_tag()` 生成可读 F-alpha 标签；`Stage1ArtifactPaths` 解析路径；三个发布函数原子写 NPZ/JSON/JSONL |
| 2 | `checkpoint.py` | `load_stage1_wrapper()` 从训练 run 的 resolved config、checkpoint 和可选代码快照恢复 wrapper |
| 3 | `full_map.py` | `infer_full_map()` 执行 80³ 滑窗物化、GPU 前向、异步 D2H 和固定顺序 Gaussian 融合 |
| 4 | `blobs.py` | `extract_probability_blobs()` 提取并稳定排序一个概率阈值下的全部 26 邻域连通区域 |
| 5 | `centered.py` | `infer_centered_boxes()` 执行候选完整前向；`pack_centered_entries()` 组装共享 offsets 的正式数组 |
| 6 | `scoring.py` | `score_centered_candidates()` 计算 basic 来源均值分数或 Find A 原子 Gaussian 分数 |
| 7 | `evaluation.py` | `semantic_prauc_histogram()` 流式累计完整图 PRAUC 事实；`aggregate_semantic_prauc()` 汇总 semantic micro/macro PRAUC；`evaluate_centered_pdb()` 计算逐 PDB 候选交集；`aggregate_stage1_metrics()` 汇总候选 micro/macro 与 top-K 指标 |
| 8 | `calibration.py` | `calibrate_semantic_thresholds()` 拟合单个 alpha 语义阈值；`tune_centered_selection()` 按固定配置顺序并行调整 basic/Gaussian 选择参数 |
| 9 | `pipeline.py` | 五个 `run_*_stage()` 直接编排跨 PDB 阶段、并行队列和文件发布 |
| 10 | `cli.py` | `main()` 解析五个子命令、JSON 清单和固定随机分片；只在 GPU 阶段构造 Dataset/wrapper |

`pipeline.py` 是生产流程的唯一编排文件，不再经过 `workflow.py`。局部嵌套函数只服务线程池回调；Dataset、collator、wrapper 和保存字段没有再被封装成顶层参数对象。

## 输出目录

一个 producer 的正常目录同时容纳生产者级调参文件、多个数据划分的逐 PDB
产物和各数据划分的评估结果：

```text
<output_root>/<producer>/
├── tuning/
│   ├── F{alpha}_semantic.json
│   ├── F{alpha}_semantic_scan.npz
│   ├── F{alpha}_basic.json
│   └── F{alpha}_gaussian.json
├── calibration/
│   ├── <pdb_id>/...
│   └── evaluation/<evaluation-name>.{jsonl,metrics.json}
├── validation/
│   ├── <pdb_id>/...
│   └── evaluation/<evaluation-name>.{jsonl,metrics.json}
└── <其他 split>/...
```

`tuning/` 只保存由完整调参清单共同产生的文件；`calibration/`、`validation/`
等数据划分目录只保存逐 PDB 目录及该数据划分的 `evaluation/`。因此
`F1_basic.json` 不会与 `calibration/9yq0/` 之类的逐 PDB 结果混放。
basic 与 Gaussian 文件按实际调参模式出现，不要求四个文件同时存在。文件不保存
checkpoint、resolved config、代码摘要或哈希。

对于 producer、数据划分和 PDB 标识，逐 PDB 根目录为：

```text
<output_root>/<producer>/<split>/<pdb_id>/
├── probability/
│   ├── probability_map.npz
│   └── geometry.json
├── blobs/F{alpha}_blobs.npz
├── centered/F{alpha}_centered.npz
├── evaluation/<evaluation-name>.npz
└── status/
    ├── probability/
    │   ├── performance.json
    │   └── _COMPLETE
    ├── F{alpha}_blobs/_COMPLETE
    └── F{alpha}_centered/
        ├── performance.json
        ├── _COMPLETE
        └── _BLOB_EXCEED
```

`F{alpha}` 使用 Python float 的最短可往返十进制：删除整数末尾 `.0`，再把小数点改成 `p`。例如 2.0 写成 `F2`，0.5 写成 `F0p5`，1.5 写成 `F1p5`；不同 Python float 不因六位格式化而碰撞。同一 PDB 可以同时保存多个 alpha 的 blobs 与 centered，并共同复用 probability。

数据划分级评估位于 `<output_root>/<producer>/<split>/evaluation/`，每个评估名称同时保存 `.jsonl` 与 `.metrics.json`。

## `probability_map.npz`

形状记号 `D/H/W` 分别是完整图 Z/Y/X 轴长度。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `probability_map` | `float32 (D,H,W)` | 不乘受体 hardmask 的配体概率，后三轴按 ZYX 排列 |
| `origin_xyz` | `float32 (3,)` | 完整网格角点的世界 XYZ 坐标，单位 Å |
| `voxel_size_xyz` | `float32 (3,)` | 世界 XYZ 体素尺寸，单位 Å/voxel |

`geometry.json` 精确字段为：`full_shape_zyx` 是长度 3 的整数列表；`origin_xyz` 与 `voxel_size_xyz` 是长度 3 的浮点数列表；`window_shape_zyx` 是固定 `[80,80,80]` 的整数列表；`stride_zyx` 是长度 3 的显式整数列表；`gaussian_sigma` 是浮点数；`window_count` 是整数。`performance.json` 的 `wall_seconds`、`materialize_wait_seconds` 与 `fusion_wait_seconds` 均为浮点秒数；计时不混入科学 NPZ。

## `F{alpha}_blobs.npz`

`N_blob` 是连通区域数量，`L_voxel` 是全部区域拼接后的体素数量。连通区域阶段不应用 `min_voxels`。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `blob_index` | `int32 (N_blob,)` | 当前文件内从 0 开始的稳定连续 blob 编号 |
| `voxel_offsets` | `int64 (N_blob+1,)` | 第 i 个区间 `[voxel_offsets[i]:voxel_offsets[i+1])` 同步切分 `voxel_index_global_zyx` 与 `source_probability`；首值 0，末值 L_voxel |
| `voxel_index_global_zyx` | `int32 (L_voxel,3)` | 完整图 ZYX 体素索引 |
| `source_probability` | `float32 (L_voxel,)` | 与完整图体素索引逐项对齐的概率 |
| `source_probability_mean` | `float32 (N_blob,)` | 每个 blob 的正式平均概率和第一排序键 |
| `voxel_count` | `int32 (N_blob,)` | 每个 blob 的体素数，等于相邻 `voxel_offsets` 之差 |
| `fits_centered_box` | `bool (N_blob,)` | blob 包围盒是否能由完整图内合法 80³ BOX 容纳 |
| `centered_box_start_zyx` | `int32 (N_blob,3)` | 可容纳时为选定 80³ BOX 的完整图 ZYX 起点；不可容纳时三个值均为 -1 |
| `source_threshold_value` | `float32 (1,)` | 当前文件使用的包含端点语义概率阈值 |

区域先按归档后的 float32 平均概率降序，再按区域最小完整图 C-order 线性索引升序。`fits_centered_box=false` 不删除 blob；basic tune 与 blobs evaluate 仍可使用该区域。

## `F{alpha}_centered.npz` 共同字段

`N` 是进入 centered 前向的候选数。候选必须同时满足 `fits_centered_box=true` 和命令显式 `forward_min_voxels`；选择参数中的 `prefiltered_min_voxel` 与 `min_voxels` 不改变该候选轴。`L_voxel` 是 N 个来源 blob 拼接后的体素数，`L_aux` 是 hardmask 内辅助受体体素拼接后的数量，`C_voxel` 是模型 `voxel_final` 特征宽度。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `centered_box_index` | `int32 (N,)` | 当前 centered 文件内从 0 开始的连续编号 |
| `source_blob_index` | `int32 (N,)` | 指向同 alpha blobs 文件 `blob_index` 第一维的编号 |
| `box_start_zyx` | `int32 (N,3)` | 当前 80³ BOX 在完整图中的 ZYX 起点 |
| `box_shape_zyx` | `uint8 (N,3)` | 每个候选固定为 `(80,80,80)` |
| `box_origin_world` | `float32 (N,3)` | 当前 BOX 角点的世界 XYZ 坐标，单位 Å |
| `voxel_size_world` | `float32 (N,3)` | 世界 XYZ 体素尺寸，单位 Å/voxel |
| `source_probability_mean` | `float32 (N,)` | 来源 blob 的完整图平均概率 |
| `source_threshold_value` | `float32 (N,)` | 来源 blobs 文件使用的语义概率阈值 |
| `voxel_offsets` | `int64 (N+1,)` | 同步切分 `voxel_index_local_zyx`、`source_probability`、`centered_probability` 和 `voxel_final`；首值 0，末值 L_voxel |
| `voxel_index_local_zyx` | `int16 (L_voxel,3)` | 来源 blob 体素在当前 80³ BOX 内的 ZYX 索引 |
| `source_probability` | `float32 (L_voxel,)` | 与来源局部体素逐项对齐的完整图概率 |
| `centered_probability` | `float32 (L_voxel,)` | 同一体素在 centered 完整前向中的重算概率 |
| `voxel_final` | `float16 (L_voxel,C_voxel)` | 与来源局部体素逐项对齐的最终 V 学习特征 |
| `voxel_aux_offsets` | `int64 (N+1,)` | 同步切分 `voxel_aux_index_local_zyx` 与 `voxel_aux_probability`；首值 0，末值 L_aux |
| `voxel_aux_index_local_zyx` | `int16 (L_aux,3)` | hardmask 内辅助受体体素的 80³ BOX-local ZYX 索引 |
| `voxel_aux_probability` | `float32 (L_aux,)` | 与辅助受体体素逐项对齐的独立预测概率，不改变配体概率 |
| `v_centroid_local_zyx` | `float32 (N,3)` | 来源 blob 在 80³ BOX 内的 ZYX 整数体素下标算术平均，单位 voxel |
| `crop_start_local_zyx` | `int16 (N,3)` | 48³ 裁块在 80³ BOX 内的 ZYX 起点 |
| `crop_center_offset_zyx` | `float32 (N,3)` | 来源 blob 质心相对 48³ 裁块中心的 ZYX 偏移，单位 voxel |
| `crop_clipped_axis_mask` | `bool (N,3)` | True 表示该轴的 48³ 起点受 80³ 边界限制 |
| `experimental_density_48` | `float32 (N,48,48,48)` | 实验密度裁块，后三轴按 ZYX 排列 |
| `simulated_density_48` | `float32 (N,48,48,48)` | 模拟密度裁块，后三轴按 ZYX 排列 |
| `source_probability_48` | `float32 (N,48,48,48)` | 完整图概率裁块，后三轴按 ZYX 排列 |
| `score` | `float32 (N,)`，条件字段 | 仅传入选择参数或执行 score-only 后存在；basic 为来源平均概率，Gaussian 为来源均值加 A 原子正负项 |
| `selected` | `bool (N,)`，条件字段 | 仅与 `score` 同时存在；True 表示同时达到分数阈值、固定 `prefiltered_min_voxel` 和最终 `min_voxels`，三个门槛均包含端点 |

不带选择参数的 centered 文件没有 `score` 和 `selected`。score-only 只能增加或替换这两个字段，其余字段、候选顺序、offsets 和数组数值保持不变。

## Find centered A/P 扩展字段

`Find_*` producer 在共同字段上增加本节 A/P 字段；`unet_*` 不产生 A 原子或 P 点字段。`N_A/N_P` 是 A 原子与 P 点总数。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `A_offsets` | `int64 (N+1,)` | 同步切分 `A_global_index`、`A_coord_local_xyz`、`A_coord_centered_world`、`A_probability` 与 `A_feat_L0/L1/L2/L3`；首值 0，末值 N_A |
| `A_global_index` | `int64 (N_A,)` | 指向 `receptor_tokens.npz` 原子轴的全局编号 |
| `A_coord_local_xyz` | `float32 (N_A,3)` | A 原子的 80³ BOX-local XYZ 体素坐标 |
| `A_coord_centered_world` | `float32 (N_A,3)` | A 原子相对 80³ BOX 世界中心的 XYZ 位移，单位 Å |
| `A_probability` | `float32 (N_A,)` | A 原子配体概率 |
| `A_feat_L0` | `float32 (N_A,50)` | 49 维 receptor token 与同原子 `is_backbone` 拼接结果 |
| `A_feat_L1/L2/L3` | `float16 (N_A,C_A*)` | 三层 A 学习特征，与 A 原子轴逐项对齐 |
| `P_offsets` | `int64 (N+1,)` | 同步切分 `P_coord_local_xyz`、`P_probability` 与 `P_feat_L2/L3`；首值 0，末值 N_P |
| `P_coord_local_xyz` | `float32 (N_P,3)` | P 点的 80³ BOX-local XYZ 体素坐标 |
| `P_probability` | `float32 (N_P,)` | P 点配体概率 |
| `P_feat_L2/L3` | `float16 (N_P,C_P*)` | 两层 P 学习特征，与 P 点轴逐项对齐 |

A 表只保留核心 80³ BOX 内且到来源 blob 最近体素中心不超过 10 Å 的原子。Gaussian 评分再使用固定 5 Å 截断。学习特征使用 float16；概率、几何、密度和 `A_feat_L0` 使用 float32。

## `_BLOB_EXCEED`

`run_centered_stage()` 在读取 blobs 后立即检查 `blob_index` 长度。长度严格大于
全局常量 1000 时总是写出：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `pdb_id` | 字符串 | 当前小写 PDB 标识 |
| `centered_role` | 字符串 | 当前动态角色，例如 `F2_centered` |
| `source_blob_count` | 整数 | 来源 blobs 文件中的总 blob 数 |
| `limit` | 整数 | 固定为 1000 |

默认行为仍是写标记后跳过当前 PDB，不产生 centered `_COMPLETE`。命令显式提供
`--continue-on-blob-exceed` 时，该文件只作为超量提示，当前 PDB 继续生成 centered
NPZ 与 `_COMPLETE`；后续 tune/evaluate 按已有 centered 产物正常读取，不检查该
提示标记。代码不删除旧标记、不建立恢复清单，也不增加独立状态机。未生成
centered 的旧式超量 PDB 仍由 tune/evaluate 在标准输出说明后跳过。

## tuning 文件

### `F{alpha}_semantic.json`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `alpha` | 浮点数 | 当前语义 F-alpha 参数 |
| `denominator` | 整数 | 阈值网格分母 |
| `pdb_count` | 整数 | 参与 PDB 等权 macro 平均的 calibration PDB 数量 |
| `positive_voxel_count` | 整数 | calibration 全集真实配体体素数 |
| `negative_voxel_count` | 整数 | calibration 全集真实背景体素数 |
| `threshold_grid_index` | 整数 | 首个达到最大 PDB 等权 macro F-alpha 的网格编号 |
| `threshold_value` | 浮点数 | `threshold_grid_index/denominator` 得到的概率阈值 |
| `macro_f_beta` | 浮点数 | 获胜阈值的 PDB 等权 macro F-alpha |
| `tp/fp/fn` | 整数 | 获胜阈值下跨 PDB 汇总的诊断体素计数，不参与阈值选择 |

每个 calibration PDB 先在同一概率网格独立计算 F-alpha；任一 PDB 的指标分母
为零时，该网格位置记为 0.0。随后对调参清单中的全部 PDB 等权平均，并用
`np.argmax` 选择首个最大值。因此并列时仍选择最低网格编号，PDB 的体素数量
不会改变其权重。

`F{alpha}_semantic_scan.npz` 保存 `denominator` int32 标量、`alpha` float64
标量、`threshold_grid_index` int32 `(denominator+1,)`、
`macro_f_beta_curve` float64 `(denominator+1,)` 和同形 int64 `tp/fp/fn`。
后三项是跨 PDB 汇总的诊断计数。

### `F{alpha}_basic.json` 与 `F{alpha}_gaussian.json`

共同字段为：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `alpha` | 浮点数 | 文件标签对应的语义 F-alpha 参数 |
| `objective` | 浮点数 | semantic macro F-beta、coverage@0.3 macro F-beta 与 one-to-one@0.3 macro F-beta 之和 |
| `objective_beta` | 浮点数 | 上述三项选择目标共同使用的 beta |
| `score_mode` | 字符串 | `basic` 或 `gaussian` |
| `score_parameters` | JSON 对象 | basic 为空对象；Gaussian 含三个下述浮点字段 |
| `score_threshold` | 浮点数 | 包含端点的候选分数下限 |
| `prefiltered_min_voxel` | 整数 | tune 开始前固定的来源 blob 体素数下限；小于该值的候选在全部参数组合中保持未入选 |
| `min_voxels` | 整数 | 包含端点的选择来源体素数下限，不改变 centered 前向集合 |
| `stages` | JSON 对象 | 保存实际执行的 basic 阈值阶段，或 Gaussian 粗搜/细搜阶段，以及最终 min_voxels 阶段 |

Gaussian `score_parameters` 精确包含 `tau_angstrom`、`lambda_positive` 与 `lambda_negative`。分数为：

$$
s = \bar{p}_{blob} + \lambda_{+}\sum_i w_i p_i - \lambda_{-}\sum_i w_i(1-p_i), \qquad w_i=\exp\left(-\frac{d_i^2}{2\tau^2}\right)
$$

$d_i$ 是 A 原子到同候选来源 blob 最近体素中心的世界距离，单位 Å；仅 $d_i\le 5$ Å 的原子参与求和。basic 分数就是 $\bar{p}_{blob}$。

三项目标保持 1:1:1 等权，共用 `objective_beta`。每一项都先在单个 PDB 内由
该 PDB 的体素、候选或 ligand occurrence 计算 F-beta，再对调参清单中的全部
PDB 取算术平均；局部分母为零时该 PDB 对应项记为 0.0。候选数、体素数和真实
occurrence 数量均不会改变 PDB 权重。

`prefiltered_min_voxel` 由 tune 命令显式提供，与最终搜索出的 `min_voxels` 独立；代码不裁剪后者的搜索列表。basic 按预过滤合格候选实际出现的 float32 来源平均概率降序扫描，只在目标值严格提升时替换阈值；非空候选的最佳目标仍为 0 时，保留高于最高分的空选择阈值。basic 的 `stages.score_threshold` 含 `objective` 与 `score_threshold`，`stages.min_voxels` 含 `objective` 与 `min_voxels`。Gaussian 的 `stages.coarse` 与 `stages.refined` 都含 `objective`、`tau_angstrom`、`lambda_positive`、`lambda_negative` 和 `score_threshold`；`stages.min_voxels` 同样只含 `objective` 与 `min_voxels`。score-only 与 evaluate 按选择 JSON 同时应用两个体素数门槛。

### Li `basic_ratio` 实验接口

`ops/stage1_li_ratio_trial/` 在独立实验中复用本目录的科学函数。它为每个 PDB
计算 Li 阈值并发布 `Li_blobs`，再以 `score_mode=basic_ratio` 搜索逐 PDB 候选
保留比例。比例总体先由 `prefiltered_min_voxel` 固定，保留数是
`floor(N*r+0.5)`；最终 `min_voxels` 只作后过滤。调参精确扫描跨 PDB 的全部
候选数变化点，目标并列时保留较小比例。选择 JSON 使用
`score_ratio_threshold`，不同时保存 `score_threshold`。

正式 Stage1 CLI 的 `basic`、`gaussian` 与动态 F-alpha 路径保持不变。
`run_evaluate_stage()` 接收显式候选角色，因此实验可以直接评估 `Li_blobs`，不把
它重命名为 F-alpha 产物。完整实验契约、字段和服务器入口见
`ops/stage1_li_ratio_trial/README.md`。

## 评估文件

每次 `evaluate` 必须显式提供 `--evaluation-name`，该名称直接决定每 PDB 的 `evaluation/<evaluation-name>.npz`、数据划分的 `evaluation/<evaluation-name>.jsonl` 和 `evaluation/<evaluation-name>.metrics.json`。不同选择参数使用不同名称即可并存；程序不对名称或参数生成摘要。

候选范围必须二选一。`--selection-parameters <json>` 按 basic 或 Gaussian 参数重算 `score` 与 `selected`；有效组合是 blobs+basic、centered+basic 和 Find centered+Gaussian。`--all-candidates` 不执行二次打分，以 `source_probability_mean` 稳定排序，并把当前 blobs 或 centered 产物中全部候选的 `selected` 设为 True。blobs 的全部候选来自 `F{alpha}_blobs.npz`；centered 的全部候选只包括已经写入 `F{alpha}_centered.npz` 的候选。

评估 NPZ 字段为：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `coverage_thresholds` | `float32 (T,)` | 双向覆盖阈值轴 |
| `topk_values` | `int32 (K,)` | top-K 数量轴 |
| `occurrence_id` | `int32 (N_gt,)` | 真实 ligand occurrence 标识轴 |
| `source_blob_index` | `int32 (N_pred,)` | 按分数稳定降序的来源 blob 编号 |
| `candidate_score` | `float32 (N_pred,)` | 与候选轴对齐；参数过滤模式保存重算分数，全候选模式保存 `source_probability_mean` |
| `candidate_selected` | `bool (N_pred,)` | 与候选轴对齐；参数过滤模式表示是否达到三个门槛，全候选模式全部为 True |
| `intersections` | `int64 (N_pred,N_gt)` | 每对候选与 occurrence 的体素交集数 |
| `pred_sizes` | `int64 (N_pred,)` | 每个候选的来源体素数 |
| `gt_sizes` | `int64 (N_gt,)` | 每个 occurrence 的体素数 |
| `candidate_semantic_tp` | `int64 (N_pred,)` | 每个候选与真实 occurrence 并集的体素交集数 |
| `semantic_tp/fp/fn` | `int64` 标量 | 已选候选体素并集的语义计数 |
| `coverage_pred_hit_mask` | `bool (T,N_pred)` | 每个完整候选是否双向覆盖至少一个 occurrence，与 selected 无关 |
| `coverage_gt_hit_mask` | `bool (T,N_gt)` | 每个 occurrence 是否被至少一个已选候选双向覆盖 |
| `one_to_one_match_offsets` | `int64 (T+1,)` | 按覆盖阈值同步切分 `one_to_one_match_pred_index` 与 `one_to_one_match_gt_index`；首值 0，末值 L_match |
| `one_to_one_match_pred_index` | `int32 (L_match,)` | 指向分数排序后完整候选轴的下标 |
| `one_to_one_match_gt_index` | `int32 (L_match,)` | 与前项逐项对齐的 occurrence 轴下标 |
| `topk_winning_candidate_rank` | `int32 (K,T)` | 已选候选序列中首个获胜名次；未命中为 -1 |
| `topk_winning_occurrence_index` | `int32 (K,T)` | 与获胜名次对齐的 occurrence 轴下标；未命中为 -1 |

数据划分 `.jsonl` 每个已评估 PDB 一条 `pdb_id` 加指标映射；`.metrics.json` 保存同一公式的跨 PDB 汇总。两者与逐 PDB NPZ 使用同一个显式 `evaluation-name`。固定键是 `pdb_count`、`semantic_tp`、`semantic_fp`、`semantic_fn`、`semantic_micro_f1`、`semantic_micro_f2`、`semantic_macro_f1`、`semantic_macro_f2`、`semantic_micro_prauc`、`semantic_macro_prauc` 与 `topk_eligible_pdb_count`。每个覆盖阈值标签 `{t}` 生成 `coverage_micro_precision_{t}`、`coverage_micro_recall_{t}`、`coverage_micro_f1_{t}`、`coverage_micro_f2_{t}`、`coverage_macro_f1_{t}`、`coverage_macro_f2_{t}`，以及同样六个 `one_to_one_*_{t}` 键。每个 top-K 值 `{k}` 与阈值标签 `{t}` 生成 `top{k}_success_count_{t}` 和 `top{k}_success_ratio_{t}`。阈值标签把小数点改为 `p`，例如 0.3 写成 `0p3`；任一分母为零时保存 0.0。

`semantic_micro_prauc` 和 `semantic_macro_prauc` 使用本次实际完成候选评估的 PDB 完整图 `probability_map` 与 `union_mask`。同一批已评估 PDB 内，改变 blobs、centered 或候选选择参数不会改变 PRAUC；centered 因默认 `_BLOB_EXCEED` 行为缺失时，该 PDB 连同候选指标一起跳过。阈值精确定义为 `t_j = torch.linspace(0,1,1024,dtype=torch.float32)[j]`，其中 `j=0,...,1023`，并以 `p >= t_j` 作为阳性预测。令该阈值的精确率和召回率为 `precision_j` 与 `recall_j`，则 `AP = sum((recall_j - recall_{j+1}) * precision_j)`，并规定 `recall_1024=0`；任一比率分母为零时该比率取 0，没有正体素时 AP 取 0。micro 先合并全部已评估 PDB 的正负体素计数再计算 AP；macro 先逐 PDB 计算 AP，再按 PDB 等权平均。

## 并行与发布

一个 PDB 内部，CPU 线程提前物化 batch，当前调用线程独占 GPU，异步 D2H 结果由单独 CPU 线程按提交顺序融合或整理。跨 PDB 时，probability 与 centered 的 NPZ 压缩分别与下一个 PDB 的 GPU 前向重叠；两个 pending 配置限制尚未发布的大数组数量。

`run_tune_stage()` 使用 `calibration.workers` 并行读取多个 PDB 的候选 NPZ 与 `ligand_area.npz`，`tune_centered_selection()` 使用同一线程数并行构造逐 PDB 事实、准备 Gaussian 原子项、计算 Gaussian 粗搜与细搜组合，以及计算最终 `min_voxels` 组合。basic 的实际分数阈值扫描保持串行，因为它按分数降序累计语义计数、覆盖状态和一对一增广匹配。全部异步任务句柄（`Future`）都按 PDB 清单或配置列表的原顺序读取，再用“目标值仅严格提升才替换”的规则选取参数；任务完成顺序不会改变包含端点、分数并列或参数并列的行为。标准输出分别记录输入加载、事实构造、basic 阈值扫描、Gaussian 原子项、粗搜索、细搜索和最终体素门槛搜索的耗时；这些运行时间不写入科学 JSON。

正式 NPZ、JSON 和 JSONL 都在最终目录写临时文件，再用 `os.replace` 原子替换。`_COMPLETE` 只在对应科学 NPZ 已替换后建立，字段是 `output_role` 和 UTC 发布时间。完成标记不保存生产身份；默认用于同阶段跳过，`--overwrite` 只撤销并重算当前阶段。代码不计算摘要或哈希，也不建立 `_valid*` 校验层。

完整命令见 `训练与运行/sh/infer/README.md`；YAML 字段见 `configs/inference/README.md`；跨项目权威字段契约见 AdaLigand `文档/规划文档/BOX-level数据契约.md`。
