# Stage1 V3 推理配置

`stage1_v3.yaml` 是所有 `unet_*` 与 `Find_*` producer 共用的通用配置。`stage1_v3_a800_24cpu.yaml` 与 `stage1_v3_a100_16cpu.yaml` 保持相同科学字段，只固定本次单卡资源的并行参数。`src.inference.cli.main()` 读取一个配置文件，再直接调用 `src.inference.pipeline` 的五个阶段入口。配置不保存 checkpoint、PDB 清单、数据划分、输出根目录或 producer 名称。

```text
stage1_v3.yaml
→ src.inference.cli.main
→ src.inference.pipeline.run_*_stage
→ full_map / blobs / centered / tuning / evaluation
```

`训练与运行/sh/infer/stage1_v3.sh` 默认读取 `stage1_v3.yaml`；环境变量 `STAGE1_INFERENCE_CONFIG` 可以指向同契约的资源配置。三模型正式入口分别选择 A800/24 CPU 与 A100/16 CPU 文件，不在命令行临时拼接并行覆盖值。

## 顶层字段

| 字段 | 类型与当前值 | 读取位置与含义 |
| --- | --- | --- |
| device | 字符串，cuda:0 | CLI 把 wrapper 移到该设备；probability 与正常 centered 的模型前向共用该设备 |
| alpha | 浮点数，2.0 | blobs、centered、tune、evaluate 没有传 --alpha 时的值；路径标签为 F2 |
| objective_beta | 浮点数，2.0 | tune 没有传 --objective-beta 时，三项 PDB 等权 macro 选择目标共同使用的 F-beta 参数 |
| blob_workers | 整数，通用/A100 为 16；A800 为 24 | run_blobs_stage() 同时处理的 PDB 数 |
| publish_workers | 整数，2 | probability 和 centered 的字段打包、NPZ 压缩与发布线程数 |
| pending_probability_pdbs | 整数，通用 3；两个资源配置为 2 | GPU 已完成但 probability 尚未发布的完整图数量上限 |
| pending_centered_pdbs | 整数，通用 3；两个资源配置为 2 | GPU 已完成但 centered 尚未打包或发布的 PDB 数量上限 |

`alpha` 与 `objective_beta` 是两条独立轴。例如，可以用 alpha=2 拟合语义阈值，同时用 objective_beta=1 调整候选选择；文件仍命名为 `F2_*`，选择 JSON 则明确保存 `objective_beta=1.0`。

## `window`

`run_probability_stage()` 把本组字段逐项传给跨 PDB 复用的 `FullMapInferenceSession`：

| 字段 | 类型与当前值 | 作用 |
| --- | --- | --- |
| stride_zyx | 三个整数，[30,30,30] | 完整图 ZYX 三轴的无 padding 80³ 滑窗步长；Python 函数不设默认值 |
| gaussian_sigma | 浮点数，0.5 | 每轴规范化到 [-1,1] 后的 Gaussian 标准差，不是体素数 |
| batch_size | 整数，通用 16；A800/24 CPU 为 32；A100/16 CPU 为 16 | 一次模型前向包含的 80³ 窗口数 |
| workers | 整数，通用 12；A800/24 CPU 为 18；A100/16 CPU 为 10 | CPU 请求物化线程数；两个资源配置都为融合、发布和主线程预留约 6 个 CPU |
| prefetch_batches | 整数，通用 12；A800/24 CPU 为 18；A100/16 CPU 为 10 | GPU 前方已经提交物化且尚未前向的有序 batch 数量上限；两个资源配置分别与 `workers` 相同，使配置中的 CPU 物化线程数能够形成同等数量的待处理批次 |
| pending_fusion_batches | 整数，通用 8；两个资源配置为 4 | GPU 后方等待 CPU 有序融合的 batch 数量上限 |
| precision | 字符串，bf16 | bf16 或 float16 开启 CUDA autocast；float32 关闭 autocast |

完整图配体概率不乘受体 hardmask。窗口按 ZYX 字典序融合，末窗贴完整图边界，Gaussian 累加保持 float32 固定顺序。

## `centered`

所有 producer 都执行完整 forward，不再配置 `voxel_only/full`、`save_voxel_final` 或 `save_dense48` 三组角色开关。字段集合只在 A/P 表上按 producer 名称前缀区分：

- `unet_*`：共同几何、来源/重算概率、`voxel_final`、辅助受体体素、V-centered 48³ 几何、实验密度 48³、模拟密度 48³ 和完整图概率 48³。
- `Find_*`：上述全部共同字段，再加 A/P 表。

| 字段 | 类型与当前值 | 作用 |
| --- | --- | --- |
| batch_size | 整数，A800/H100是8，A100是4 | 一次完整 forward 的 centered 80³ BOX 数 |
| workers | 整数，通用 8；A800/24 CPU 为 18；A100/16 CPU 为 10 | CPU centered 单请求物化线程数 |
| prefetch_batches | 整数，3 | GPU 前方准备的 centered batch 数量上限 |
| pending_cpu_batches | 整数，3 | GPU 后方等待字段整理的 batch 数量上限 |
| precision | 字符串，bf16 | bf16 或 float16 开启 CUDA autocast；float32 关闭 autocast |

`forward_min_voxels` 不在 YAML 中隐藏，由每次 `centered` 命令显式提供。来源 blob 总数上限固定为代码常量 1000，配置中不再暴露第二份可变值。

## `calibration`

配置节继续命名为 `calibration`，因为这些网格只在 calibration PDB 清单上拟合；
拟合结果则写入 producer 的 `tuning/`，不与 `calibration/<pdb_id>/` 混放。正式
调参固定使用 PDB 等权 macro，不提供 micro/macro 模式开关。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| workers | 整数，通用/A100 为 16；A800 为 24 | tune 阶段并行加载候选与 occurrence 文件、构造逐 PDB 事实、归约 Gaussian 原子项并计算相互独立的参数组合 |
| semantic_denominator | 整数 | 语义概率网格分母；网格编号 j 对应包含端点的阈值 j/denominator |
| min_voxel_values | 整数列表 | basic 与 Gaussian 最后阶段依次扫描的来源 blob 最小体素数 |
| gaussian_grid.tau_angstrom | 浮点数列表 | Gaussian 距离标准差粗网格，单位 Å |
| gaussian_grid.lambda_positive | 浮点数列表 | A 原子正项系数粗网格 |
| gaussian_grid.lambda_negative | 浮点数列表 | A 原子负项系数粗网格 |
| gaussian_refinement.lambda | 浮点数列表 | 分别乘到粗搜索最优正、负系数的细搜索乘数 |

两种 tune 模式都要求命令显式提供 `prefiltered_min_voxel`。该值在任何参数尝试前固定，体素数不足的候选始终未入选，但 PDB 与真实 occurrence 仍保留在评估事实中。basic 随后按合格候选实际出现的 float32 来源平均概率降序串行扫描，因为每个阈值都在前一个阈值的累计候选与一对一匹配状态上继续更新；该扫描只在目标值严格提升时替换阈值，非空候选的最佳目标仍为 0 时保留高于最高分的空选择阈值。Gaussian 对粗网格中的每组 tau 和两个 lambda 精确扫描实际 float32 候选分数，第二阶段固定 tau、细化两个 lambda 并再次精确扫描分数，最后冻结分数后搜索 `min_voxel_values`。事实构造、Gaussian 原子项、相互独立的参数组合和最终体素门槛目标由 `workers` 外层并行；结果始终按配置原顺序读取后再执行严格提升规则，因此线程完成顺序不改变端点语义或参数并列时的首项获胜。`stage1_v3.sh` 继续把 OMP、MKL 与 OpenBLAS 的内部线程数固定为 1，避免外层并发再嵌套计算线程。

## `evaluation`

| 字段 | 类型与当前值 | 作用 |
| --- | --- | --- |
| coverage_thresholds | 浮点数列表，[0.3,0.5,0.6] | 逐 PDB 评估 NPZ 的双向覆盖阈值轴；调参目标固定使用 0.3 |
| topk_values | 整数列表，[3,4,5] | top-K 成功指标的 K 轴 |

评估仍报告 semantic、双向 coverage、一对一最大匹配、top-K、跨 PDB micro 与 PDB 等权 macro 指标。语义指标固定发布 `semantic_micro_prauc` 和 `semantic_macro_prauc`；每个 coverage 阈值还固定发布 coverage 与 one-to-one 的 micro/macro PRAUC。候选 PRAUC 先应用两个固定体素数门槛，忽略最终分数阈值，再扫描实际 float32 候选分数。精确公式见 `src/inference/README.md`。这些口径不是运行参数，YAML 不增加开关。YAML 不控制输出文件名；`evaluate` 命令必须显式提供 `--evaluation-name`，并在 `--selection-parameters` 与 `--all-candidates` 之间二选一。前者支持 blobs+basic、centered+basic 和 Find centered+Gaussian；Gaussian 需要 centered A 原子字段，不能用于 blobs。后者不做二次打分，把 `--artifact` 指定候选文件中的全部候选纳入指标，并用 `source_probability_mean` 排序。

## 版本目录与复用

同一 `output_root` 可以复用 probability 并逐次增加 `F1`、`F2`、`F3` 或小数 alpha 产物。代码不计算 checkpoint、配置或代码摘要，也不要求一个 checkpoint 只能对应一个结果目录。若调用者需要区分同一 checkpoint 的 F3 最优、F2 最优或其他科学配置，使用易读的目录名称区分即可。

修改 YAML 后，正式 release/launch 记录应保存实际文件。解析验证使用项目的 OmegaConf 环境：

```powershell
$prefix = (conda env list --json | ConvertFrom-Json).envs | Where-Object { [System.IO.Path]::GetFileName($_) -eq "Pocket_Plus_windows" } | Select-Object -First 1
& (Join-Path $prefix "python.exe") -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/inference/stage1_v3.yaml'); OmegaConf.resolve(c)"
```
