# Stage1 V3 推理配置

`stage1_v3.yaml` 是 `unet_c1`、`Find_0`、`Find_1` 和 `Find_2` 共用的科学与并行配置。producer、checkpoint、训练 resolved config、PDB 清单、数据划分和输出根目录由 `src.inference.cli` 显式传入，本目录不保存部署路径。

读取链路：

```text
stage1_v3.yaml
→ src.inference.cli.main
→ src.inference.workflow
→ src.inference.pipeline
→ full_map / blobs / centered / calibration / evaluation
```

## 顶层执行字段

| 字段 | 读取位置 | 作用 |
| --- | --- | --- |
| `device` | `cli.py` | 唯一 GPU owner 的设备 |
| `blob_workers` | `workflow.py` | 跨 PDB 26 邻域连通区域线程数 |
| `publish_workers` | `workflow.py` | NPZ 压缩和 JSON 发布线程数 |
| `pending_probability_pdbs` | `workflow.py` | 尚未完成概率/ blobs 发布的完整图上限 |
| `pending_centered_pdbs` | `workflow.py` | 尚未完成 centered 发布的 PDB 上限 |

正式运行总是为当前 checkpoint 和配置重新计算概率图。PDB 角色 `_COMPLETE` 不保存生产身份，因此不能作为跨 checkpoint 复用依据。

## `window`

`stride_zyx`、`gaussian_sigma`、`batch_size`、`workers`、`prefetch_batches`、`pending_fusion_batches` 和 `precision` 由 `pipeline.py` 逐项传给 `infer_full_map()`。`stride_zyx` 没有 Python 默认值；当前建议配置显式使用 `[50,50,50]`。`gaussian_sigma=0.5` 属于每轴归一化到 `[-1,1]` 的坐标，不是体素数。

## `centered_common` 与角色覆盖

YAML anchor 只复用 centered batch、物化线程、预取深度、CPU 整理队列和精度。每个角色仍显式声明：

- `semantic_role`、blob 文件角色和 centered 文件角色；
- `min_voxel_values` 与 `objective_beta`；
- 四个 producer 对应的 `score_mode`；
- F3 blob 上限；超量只写事实并继续生产 centered；
- `forward`、`save_voxel_final` 和 `save_dense48`。

`F1_basic` 使用 voxel-only 前向，关闭 V 特征、A/P 和 48³ 数组。`F3_centered` 使用完整前向，打开 V 特征和 48³ 数组；只有 Find producer 产生 A/P 表。两者都不乘受体 hardmask。

## `calibration`

`semantic_denominator` 控制完整语义阈值网格。`gaussian_refinement.lambda` 的 5 个乘数分别应用于第一阶段最优正负 lambda；`gaussian_refinement.score_threshold` 的 15 个乘数应用于第一阶段最优 `gauss_score_min`，因此第二阶段恰有 `5×5×15=375` 组。

F3 `score_parameter_grid` 是第一阶段粗网格：5 个 tau、5 个正 lambda、5 个负 lambda 和 4 个分数下限，共 500 组。U-Net 与 F1 basic 的 `score_parameter_grid: null` 表示只使用来源平均概率，不读取 Gaussian 参数。

## `evaluation`

`coverage_thresholds` 的顺序同时决定评估 NPZ 的阈值轴和 metrics 字段；当前是 0.3、0.5、0.6。`topk_values` 的顺序决定 top-K 轴；当前是 3、4、5。校准目标固定取第一项 0.3，配置顺序不得随意改变。

修改正式配置后，release/launch 记录必须保存该文件。同一 checkpoint 使用不同推理配置时，必须传入不同的 `output_root` 版本目录；代码不计算配置摘要，也不自动生成版本名。一个版本目录内的 calibration、validation 和 train 使用同一份配置。
