# easy20 oracle / realistic docking 实验记录

本文档记录本轮降低标准验证实验的配置、运行状态、结果和分析。它面向用户阅读，也作为后续 AI agent 接手时的实验说明。本文中的“成功”只在明确写成 RMSD、assignment accuracy 或 center hit 时才表示科学意义上的成功；Rosetta 进程结束只叫“流程跑通”。

## 实验目的

本轮实验回答两个问题。

第一，如果给分子对接工具更好的初始信息，Rosetta / 当前 pipeline 能不能把 ligand pose 做到合理 RMSD。这条路线用于区分“前置网络中心不好”与“下游 docking 本身不好”。

第二，从 80 个样本里筛出两套快速验证集合，后续任何修复或新打分方法都能先在小集合上快速迭代。

## easy20 样本定义

`easy20_by_nk` 是计算量最少的 20 个样本。这里：

- `n` 表示去掉纯金属离子后的 dockable ligand occurrence 数量。
- `k` 表示保守后处理后进入 docking 的预测 site 数量。
- `nk = n * k`，近似代表真实流程里单样本 site-ligand 矩阵规模。

`easy20_by_instance_f1` 是前置预测最好的 20 个样本。排序指标为推理管线逐样本的 `instance_f1`。这个指标来自前置网络的 instance/连通域评估，不是 docking pose RMSD，也不是 center hit。

## 保守 instance 后处理

本轮统计 `k` 时使用保守后处理：

- `min_voxels = 1`：不额外去除小体素 instance。
- `merge_center_distance = 8.0 Å`：只允许中心距离小于 8 Å 的原始 instance 合并。
- `merge_min_voxel_distance` 和 `merge_max_bbox_span_increase` 放宽为很大值，使这次合并主要由中心距离控制。

这个后处理只服务于 easy20 统计和快速验证，不等同于最终推理管线参数。

## ligand 分层统计

每个样本会记录：

- `num_dockable_ligands`：排除纯金属离子后的可对接 ligand 数。
- `num_glycan_like_ligands`：CCD ID 明确属于糖类或支链糖类的 ligand 数。
- `bad_at_tool_ratio_conservative`：`num_glycan_like_ligands / num_dockable_ligands`。这是保守版“不擅长 ligand 比例”，当前只把明确糖类计入。
- `num_large_ligands`：heavy atom 数大于 80 的 ligand 数。
- `num_internal_metal_ligands`：ligand 内部含金属元素的数量；它不等于纯金属离子，纯金属离子已在 dockable ligand 外排除。
- 前置网络指标：`instance_precision`、`instance_recall`、`instance_f1`、`voxel_precision`、`voxel_recall`、`voxel_f1`。

## 四类 oracle 实验

四类实验均在 `easy20_by_nk` 上运行，receptor 使用 `true_receptor` 和 `cryoatom_receptor` 两套。

### true_center_identity

使用真实 ligand 几何中心作为 docking 初始中心，并且每个真实中心只对接对应的真实 ligand 身份。这个实验是下游 docking 的最宽松上限测试：中心和 ligand 身份都告诉工具，只看 Rosetta 是否能 refine 到正确 pose。

### offset_center_identity

使用真实 ligand 几何中心加随机偏移作为 docking 初始中心，并且仍然只对接对应真实 ligand 身份。偏移半径为 4/6/8 Å，每个半径 3 个 seed。这个实验测试“中心有误差但 ligand 身份正确”时，下游 docking 的容错范围。

### true_center_hungarian

使用每个真实 ligand 几何中心作为 site，但不告诉 site 对应哪个 ligand；每个 site 与候选 ligand 都跑 docking，再根据分数做 Hungarian matching。这个实验测试“中心正确，但 ligand 身份需要靠 scoring/matching 判断”。

### offset_center_hungarian

使用真实中心加 4/6/8 Å 球内均匀偏移，每个半径 3 个 seed；每个 offset variant 独立做 Hungarian matching。这个实验同时测试中心误差和 ligand identity ranking/matching。

## realistic 实验

`easy20_by_instance_f1` 上运行真实流程 docking：使用前置网络预测 site、保守后处理、候选 ligand、`true_receptor` 和 `cryoatom_receptor`，再做当前 pipeline 的 assignment。它回答“当前真实设置下，前置预测较好的样本能否得到更好的 docking 结果”。

## Rosetta 与资源配置

- 默认 `nstruct = 5`。
- 若 easy20 统计显示 oracle identity 子集 job 数非常小，可在后续单独把 identity 子集提高到 `nstruct = 10`。
- offset 半径：4 Å、6 Å、8 Å。
- 每个半径 seed 数：3。
- CPU 上限：96 核。
- 服务器写入范围：`/home/penghongen/分子对接尝试`。
- 本地可编辑范围：`C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking`。

## 输出位置

预扫描输出：

```text
/home/penghongen/分子对接尝试/easy20_prescans/20260521_easy20_prescan_v1
```

当前有效预扫描输出为：

```text
/home/penghongen/分子对接尝试/easy20_prescans/20260521_easy20_prescan_v6
```

oracle 输出：

```text
/home/penghongen/分子对接尝试/pipeline_runs/20260521_oracle_easy20_nk_nstruct5_array8_v1
```

realistic 输出：

```text
/home/penghongen/分子对接尝试/pipeline_runs/20260521_realistic_easy20_instancef1_nstruct5_array8_v1
```

## 当前状态

- 2026-05-21：已确认四类 oracle 任务定义，并在本地新增预扫描脚本、oracle runner、realistic sbatch 和本说明文档。
- 2026-05-21：已使用固定同步脚本同步到服务器，并提交 prescan job `274525`。该 job 5 分钟仍无表输出，经判断是中心距离合并参数触发了不必要的最近体素距离计算。该 job 由 Codex 提交，已取消。
- 2026-05-21：提交修正版 prescan job `274528`，但 3 分钟仍无表输出；进一步判断瓶颈是 prescan 不应读取完整 `instance_label`。该 job 由 Codex 提交，已取消；本地已把 prescan 改为只基于 `voxel_candidates.json` center 做 union-find 合并，待同步后提交 v3。
- 2026-05-21：提交 center-only prescan job `274531`，2 秒失败，错误为漏导入 `numpy`。本地已补 `import numpy as np`，待同步后提交 v4。
- 2026-05-21：提交 prescan job `274532`，3 分钟仍无表输出；判断重复扫描 ligand mapping CSV 仍太慢。该 job 由 Codex 提交，已取消；本地已改为一次性读取并缓存 mapping，待同步后提交 v5。
- 2026-05-21：提交 prescan job `274535`，1 分 14 秒完成，但 `instance_f1` 全为空。原因是 `per_sample_best_metrics.json` 使用 `sample_name` 字段，本地已修正指标读取，待同步后提交 v6。
- 2026-05-21：已用固定同步脚本同步 `sample_name` 修复，并提交 prescan v6 job `274539`，`PRESCAN_RUN_ID=20260521_easy20_prescan_v6`。
- 2026-05-21：prescan v6 8 秒完成，`instance_f1` 已正确读入。已提交 oracle job `274540`，并提交依赖它的 realistic job `274541`；realistic 使用 `afterany:274540`，因此不会与 oracle 同时占用 96 核。
- 2026-05-21：用户指出 96 核单 job 很难排队；已取消 Codex 提交且尚未运行的 `274540/274541`，改为 8 核 array 分片。当前任务链为 oracle array `274549`、oracle 汇总 `274550`、realistic array `274551`、realistic 汇总 `274552`。array 配置为每个 task 8 核、最多 12 个 task 并发，总 CPU 不超过 96。
- 2026-05-21：只读 SSH 检查（10.102.33.220:10022）：`274549` 当前 RUNNING（`274549_0` 到 `274549_11` 共 12 个 array task 运行中；`274549_12` 到 `274549_19` 因 `JobArrayTaskLimit` 等待）。`274550/274551/274552` 仍为 dependency 等待。
- 2026-05-21：oracle array 当前正在跑的样本（12 个）：`7n70, 7nnl, 7tju, 7vla, 7zdf, 8dd7, 8p71, 8pmd, 8v6v, 8v7l, 8wpf, 8x9s`。
- 2026-05-21：已生成审计：`samples/<sample>/audit/summary.json` 计数为 0；但 `samples/**/audit/summary.json`（包含 `variants/...` 子目录）已生成 49 个 summary，说明当前审计主要落在 variant 级目录而不是 sample 根目录（后续汇总应以 `samples/**/audit/summary.json` 为准）。
- 2026-05-21T18:11+08:00：再次尝试本轮只读检查 `squeue/sacct`、审计计数和 realistic 是否已启动；但当前工作站到 `10.102.33.220:10022` 的 SSH 在连接前即返回 `Permission denied`，且本地 `C:\Users\15919\.ssh` 目录不可读，因此本轮没有比 17:21+08:00 更新的服务器状态。不要把下面的运行中样本列表和 49 个 variant summary 当作 18:11+08:00 的新鲜观察。
- 2026-05-21：SSH 已恢复并获得新反馈。`274549` 已完成 6/20 个样本，0 个失败 task；完成样本为 `8dd7, 8x9s, 7vla, 8v7l, 8p71, 8wpf`。这些样本共 240 个 Rosetta job，240 个成功。`274550/274551/274552` 仍在 dependency pending。
- 2026-05-22：oracle array `274549` 已完成 12/20 个样本，0 个 Slurm 失败 task；8 个样本仍在运行，realistic array 仍未开始。已完成样本的样本级 summary 合计 1120/1120 个 Rosetta job 成功；但 `6bk8` 运行中的部分 variant 已暴露 Rosetta job 级失败，典型错误为 `pdb_UNK` 缺少 RamaPrePro 主链 score table。

## 结果与分析

### prescan v6

`num_samples = 80`，`num_usable_samples = 60`。

`easy20_by_nk`：

```text
8dd7, 8x9s, 7vla, 8v7l, 7zdf, 8v6v, 8p71, 8wpf, 8pmd, 7n70,
7nnl, 7tju, 7zdl, 8umt, 8wox, 8vm0, 6bk8, 7ut7, 8x0b, 7v19
```

`easy20_by_instance_f1`：

```text
8dd7, 7zdf, 8pmd, 8wzj, 8v6v, 8bly, 8vcj, 7tju, 8ut3, 7ut7,
7zdl, 8umt, 8vm0, 8wis, 8ca3, 6bk8, 8x2l, 7z7s, 8x0b, 8xh9
```

初步观察：

- `easy20_by_nk` 的前几个样本非常适合 smoke / debug：例如 `8dd7`、`8x9s` 的 `nk=1`，`7vla`、`8v7l` 的 `nk=2`。
- `easy20_by_instance_f1` 确实选到了前置网络很好的样本，但其中也包含 ligand 数或糖类比例很高的复杂样本，例如 `8ut3`、`8wis`、`8xh9`。因此 realistic 结果必须按 ligand 数、糖类比例、内部金属和 `instance_f1` 分层解释。
- `8wox` 在 `easy20_by_nk` 中 `bad_at_tool_ratio_conservative=1.0`，说明“计算量小”并不等于“化学上容易”；它适合暴露糖类/工具短板，但不应代表一般小样本。

### 调度策略调整

本轮 CPU docking 不再使用“一个 job 申请 96 核”的方式。新的 array 策略是：

- 每个样本一个 array task。
- 每个 task 申请 8 核。
- array 并发上限为 `%12`，所以最多同时 12 个 task、总计 96 核。
- 每个 task 使用 `--shard-id` 写入独立 shard summary，避免多个 task 同时覆盖同一个 `batch_summary.json`。
- array 完成后运行 2 核汇总 job，把样本级 `audit/summary.json` 合并成总 `tables/batch_summary.json`。

### oracle array 中途反馈

当前不是最终结果，只是运行健康度检查。

已完成样本：

```text
8dd7, 8x9s, 7vla, 8v7l, 8p71, 8wpf
```

这 6 个样本全部 `status=ok`。合计 120 个 variant、240 个 Rosetta job，其中 240 个成功，流程跑通率为 100%。目前没有非空 stderr 日志。

当前仍在运行的样本：

```text
7zdf, 8v6v, 8pmd, 7n70, 7nnl, 7tju, 7zdl, 8umt, 8wox, 8vm0, 6bk8, 7ut7
```

尚未启动的样本：

```text
8x0b, 7v19
```

realistic array 尚未开始；它会在 oracle array 和 oracle 汇总之后启动。

### 2026-05-22 oracle 长尾检查

已完成样本：

```text
8dd7, 8x9s, 7vla, 8v7l, 7zdf, 8v6v, 8p71, 8wpf, 8pmd, 7n70, 7zdl, 8wox
```

这些样本对应 240 个 variant、1120 个 Rosetta job，目前样本级 summary 中 1120 个成功。`tables/batch_summary.json` 仍不存在，因为 oracle 汇总 job `274550` 还在 dependency pending。

仍在运行样本：

```text
7nnl, 7tju, 8umt, 8vm0, 6bk8, 7ut7, 8x0b, 7v19
```

目前没有 Slurm 失败 task，也没有 sbatch stderr。需要注意的是，`6bk8` 的部分 variant summary 已经出现 Rosetta job 级失败。典型错误是：

```text
ERROR: Error in core::scoring::RamaPrePro::get_mainchain_torsions_covered():
No mainchain score table for residue type pdb_UNK exists.
```

这表示 array 任务本身没有失败，但 Rosetta 在处理某些输入结构时把某些 residue 当作 `pdb_UNK`，并在 Cartesian/minimization 相关 RamaPrePro 打分项上退出。后续最终汇总时应把它归为“Rosetta 输入/residue type 兼容性问题”，不能算作 Slurm 或 array 调度失败。

### ETA 粗略预测

oracle array 当前被最后 8 个样本支配，尤其 `7v19`、`7ut7`、`8x0b`、`8vm0` 这类 ligand 数较多的样本。按当前 variant 级进度线性外推，oracle 还可能需要约 12 到 36 小时；最乐观是若后续 offset Hungarian 很快失败或跳过，约半天内结束；保守估计是 `7v19` 继续长尾，约 1 到 1.5 天。

realistic array 尚未开始。它不是 oracle 的 20 个 variant 矩阵，但 `easy20_by_instance_f1` 里包含 `8ut3`、`8wis`、`8xh9` 这类复杂样本，因此仍可能有长尾。当前粗略预计全链路（oracle 汇总 + realistic + realistic 汇总）还需要约 2 到 3 天。
