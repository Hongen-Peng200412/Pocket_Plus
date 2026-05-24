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
- `--rosetta-jobs N`：一个样本内部最多同时运行的 Rosetta 子进程数；每个子进程仍对应一次独立的 ligand-site-receptor docking job。该参数对 true/offset、identity/Hungarian 产生的 jobs 一视同仁。
- 推荐调度规则：先计算该样本的计划 Rosetta job 数，再以 `ceil(jobs / 40)` 作为建议 CPU/`rosetta_jobs`；若资源宽裕且目标是快速消除长尾，可参考 `ceil(jobs / 30)`。这是可解释的经验规则，不是强制约束；所有 Codex 提交任务的总申请仍不得超过 96 CPU。

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
/home/penghongen/分子对接尝试/pipeline_runs/20260523_realistic_easy20_instancef1_nstruct5_array1_fixed_v1
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
- 2026-05-23：发现当前 8 核 array task 实际仅按约 1 个活跃 CPU 核运行 Rosetta jobs。这是资源申请性能 bug；后续新提交脚本已改为每 task 申请 1 核，当前正在运行的旧 job 不受脚本更新影响。
- 2026-05-23：为不等待 oracle 长尾而曾启动 realistic 并行任务 `280768/280769`；随后确认其也沿用“每 task 申请 8 核但仅约 1 核工作”的性能 bug，已在早期由 Codex 取消。分析不使用该 superseded partial run。
- 2026-05-23：`8x0b` 已完成；`7v19` 因 oracle 完整成本显著更大且预计仍需数日，已只取消 Codex 提交的 array task `274549_19`，保留 partial 输出。当前完整分析执行集定义为 `easy19_fast_validation = easy20_by_nk - {7v19}`；oracle 仅剩 `7ut7` 继续运行。
- 2026-05-23：修复后的 realistic run 已提交：array job `281985` / run_id `20260523_realistic_easy20_instancef1_nstruct5_array1_fixed_v1`，汇总 job `281986`。20 个样本 task 均以 1 CPU 同时开始运行；加上仍使用旧申请的 oracle `7ut7`，当前总申请 CPU 为 28，不超过 96。
- 2026-05-23T21:03+08:00：修复后的 realistic run 已完整完成 5/20 个样本：`8dd7, 7zdf, 8pmd, 8v6v, 7zdl`，合计 `60/60` 个 Rosetta jobs 跑通。当前仍只能称为流程跑通，需待 assignment 与 RMSD 评估后才能讨论 docking 质量。
- 2026-05-23：用户要求实现可复用的样本内部并行，而不是将 `7v19` 永久剔除。代码已加入公共 `run_rosetta_jobs(..., rosetta_jobs)` 调度层；oracle 会把一个样本的所有 task/variant 产生的 Rosetta jobs 放入同一池，realistic 也复用相同执行器。下一步是在原 oracle run 中用 24 CPU 全量覆盖 `7v19` 并观察实际加速。
- 2026-05-23T21:47+08:00：首次 `7v19` 覆盖 job `283182` 在 25 秒内停止，未运行 Rosetta 对接矩阵。原因是原 partial 输出仍含该样本的 `params/*.params` 与 `*_0001.pdb`，全量重跑重新执行 `molfile_to_params.py` 时失败。下一次将只删除已获准覆盖的 `samples/7v19` 目录后重提；批处理入口也已修复为样本失败时向 Slurm 返回非零状态。
- 2026-05-23T21:52+08:00：已只清理原 run 的 `samples/7v19` 并提交新的 24 核全量覆盖 job `283183`。首次检查时 job 已在 `cnode01` 运行且重建了 899 个文件，证明参数准备冲突已消除；是否达到实际并行加速仍待 CPU/输出监视确认。
- 2026-05-23T21:53+08:00：`283183` 运行 `1:37` 后，Slurm `sstat` 显示累计 CPU 时间 `17:28`，即约 `10.8` 个墙钟核同时有效工作；这已经证明 `--rosetta-jobs=24` 不再是空申请。此时首批对接尚未完成，scorefile 计数仍为 0 且 stderr 为空；将再观察一次落盘结果后启动 `7ut7` 的并行覆盖。
- 2026-05-23T21:56+08:00：`283183` 在 `6:26` 时累计 CPU 时间已达 `2:12:50`（约 `20.6x`），stderr 与样本 error 仍为空；节点约 2 TB 内存且空闲约 1.87 TB，因此该观察足以判定样本内部并行有效。已取消旧串行 `7ut7` task `274549_17` 及旧 oracle 汇总 `274550`，清空仅该样本旧输出并提交 16 核覆盖 job `283281`；新最终汇总 job `283282` 等待 `283183` 和 `283281` 后运行。最终 oracle 集合恢复为完整 easy20。
- 2026-05-23T22:02+08:00：输出验证完成。`7v19` job `283183` 在 `9:32` 已累计使用 `3:27:10` CPU（约 `21.7x`）并产生 24 个 scorefile / 76 个输出文件；`7ut7` job `283281` 在 `1:51` 已累计使用 `17:32` CPU（约 `9.5x`）。两个 stderr 均为空。这说明通用样本内并行既实际占用多核，也能正常写回原 easy20 样本目录。
- 2026-05-23：最终 collector `283282` 的样本结果读取 `samples/*/audit/summary.json`，因此同一个样本的覆盖输出只计一次；失败验证 `283182` 留下的 shard summary 只增加审计字段 `num_shards`，不会把 `7v19` 重复算入 jobs、流程跑通率或后续评价表。
- 2026-05-24T12:02+08:00：oracle 并行覆盖与 collector 均已完成。`283183` (`7v19`, 24 CPU) wall time 为 `10:29:41`，`283281` (`7ut7`, 16 CPU) wall time 为 `04:50:59`，`283282` 已写出完整 easy20 `tables/batch_summary.json`。
- 2026-05-24T12:02+08:00：realistic `281985` 已完整完成 12/20 个样本、636/636 个 Rosetta jobs 流程跑通；尚有 8 个串行样本未完成，collector `281986` 仍等待依赖。

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

本轮已运行的 CPU docking 不再使用“一个 job 申请 96 核”的方式，而采用了如下 array 策略：

- 每个样本一个 array task。
- 每个 task 申请 8 核。
- array 并发上限为 `%12`，所以最多同时 12 个 task、总计 96 核。
- 每个 task 使用 `--shard-id` 写入独立 shard summary，避免多个 task 同时覆盖同一个 `batch_summary.json`。
- array 完成后运行 2 核汇总 job，把样本级 `audit/summary.json` 合并成总 `tables/batch_summary.json`。

2026-05-23 发现上述配置中存在严重性能 bug：runner 在一个 task 内顺序执行 Rosetta，且 `JOBS=1`、各数值线程变量均为 1。Slurm 检查显示例如 `8dd7` 的 wall time 为 `6971 s`，`TotalCPU=6931 s`，说明申请的 8 核中只有约 1 核真正工作。对已有串行 array，资源配置已修正为每 task 申请 1 CPU。随后根据用户决定，代码进一步加入 `--rosetta-jobs`：当单个样本具有大量独立 docking jobs 时，可以在该样本内真实并发运行 Rosetta，而不是白白申请多核。

新的通用执行方式不区分 job 来自哪一种试验定义：`identity` 表示后续无需由打分选择 ligand，`Hungarian` 表示对接结果随后用于匹配；二者的单次 Rosetta 计算均是独立 ligand-site-receptor job，都进入相同执行池。并行时 Rosetta 子进程的工作目录使用各自独占输出目录，避免失败日志 `ROSETTA_CRASH.log` 在共享目录互相覆盖。

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

旧的串行估计说明了为什么必须修复并行：`7v19` 若按原串行方式完成剩余部分需要约 `105.9` 小时（P75）到 `125.0` 小时（P90）。现在 `7v19` 已由 `283183` 以 24 核运行全部 `1440` jobs；串行全量 P75/P90 成本约 `132.4/156.3` CPU-core 小时，24 路理想 wall time 约 `5.52/6.51` 小时，真实监视已看到约 `21.7x` CPU/墙钟比与首批 scorefile 正常落盘，因此仍以从提交起 `6-10` 小时作为含准备与长尾开销的保守预测。`7ut7` 已改由 `283281` 以 16 核全量覆盖；其 `840` jobs 的串行 P75/P90 成本约 `77.2/91.1` CPU-core 小时，16 路理想 wall time 约 `4.83/5.69` 小时，并已观察到约 `9.5x` 初始 CPU/墙钟比，保守先按 `5-9` 小时预测。最终汇总 `283282` 会在两者收尾后恢复完整 easy20 结果。

realistic 的低效旧 array `280768` 已取消；修复后的 1-CPU array `281985` 已在 oracle 尚未完全结束时启动。约 2.6 小时后它已完成 5/20 个样本和 `60/60` 个 Rosetta jobs，说明修复后的资源配置可以正常推进输出；后续将用同一单位成本口径继续估算长尾样本剩余时间。

### 2026-05-23 时间预算速查

本节的目的不是列出每个输入细节，而是让后续 AI agent 可以立刻估算新实验需要多久。一次 `Rosetta docking job` 指：在 `nstruct=5` 下，将一个 ligand 放到一个给定中心并对一个 receptor 运行一次 Rosetta 的子进程。当前 runner 为串行、单活跃 CPU 核执行，因此下表中的秒数可以作为后续 1 CPU task 的 wall-time 预算。

服务器结构化输出位于：

```text
/home/penghongen/分子对接尝试/pipeline_runs/20260521_oracle_easy20_nk_nstruct5_array8_v1/tables/runtime_budget_v1
```

| 口径 | 数值 | 使用方法 |
| --- | ---: | --- |
| 完整样本数 | 18 | 生成报告时已有 18 个完整 oracle 样本 |
| 成功 jobs | 3459 | 用于正常运行时间预算 |
| 失败 jobs | 61 | 单列实际浪费，不混入正常预算 |
| 成功 job Median | 220.3 秒 / 3.67 分钟 | 描述典型 job |
| 成功 job P75 | 331.1 秒 / 5.52 分钟 | 新任务默认估算 |
| 成功 job P90 | 390.6 秒 / 6.51 分钟 | 长尾保守上界 |
| 失败 job 已消耗 | 3.88 CPU-core 小时 | 主要来自 `6bk8` 的 Rosetta 输入兼容问题 |

本轮 oracle 不应使用 prescan 的预测 `nk` 直接估算成本，因为它使用真实中心与真实 ligand 集合。令 `n` 为真实 dockable ligand 数，当前四类任务和两个 receptor 合计产生：

```text
完整 oracle jobs = 20 * n * (n + 1)
默认 wall-time 预算 = jobs * 331.1 秒
保守 wall-time 上界 = jobs * 390.6 秒
```

| 样本 | n | prescan nk | 完整 oracle jobs | 状态 | 已实测耗时 | P75 剩余预算 | P90 剩余预算 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `7ut7` | 6 | 36 | 840 | `283281` 等待/运行 16 核全量覆盖 | 旧串行 partial 不再作为最终结果 | 串行全量 77.25 小时；16 路理想 4.83 小时 | 串行全量 91.15 小时；16 路理想 5.70 小时 |
| `8x0b` | 6 | 36 | 840 | 已完成 | 46.87 小时 | 0 | 0 |
| `7v19` | 8 | 48 | 1440 | 等待 24 核全量覆盖并行验证；旧 partial 保留至覆盖开始 | 37.44 小时 partial | 串行全量 132.42 小时；24 路理想 5.52 小时 | 串行全量 156.25 小时；24 路理想 6.51 小时 |

因此 `7v19` 不是在 `nk` 几乎相同条件下凭空变慢：它的真实 ligand 数从 6 增到 8，使本轮 oracle 的完整 job 数从 840 增到 1440。样本内部并行已经在其真实运行中显示有效，`7v19` 与 `7ut7` 都将回到最终 easy20 汇总；`easy19_fast_validation` 不再是结果集合，只是一段已经被修复行动替代的诊断历史。

### 2026-05-24 oracle 完整 easy20 流程汇总

最终 oracle 表已经可读，但本节统计的是流程能否完成及 assignment 审计字段是否存在，不是最终几何正确率。

| 范围 | Rosetta jobs | 流程跑通 jobs | 跑通率 |
| --- | ---: | ---: | ---: |
| 全部 oracle easy20 | 5800 | 5739 | 98.95% |
| `true_center_identity` | 112 | 110 | 98.21% |
| `offset_center_identity` | 1008 | 990 | 98.21% |
| `true_center_hungarian` | 468 | 463 | 98.93% |
| `offset_center_hungarian` | 4212 | 4176 | 99.15% |
| `true_receptor` | 2900 | 2840 | 97.93% |
| `cryoatom_receptor` | 2900 | 2899 | 99.97% |

完整集合中 20/20 个样本均生成 summary，且无样本级 skip；61 个失败 Rosetta jobs 全部集中在 `6bk8`，该样本为 `59/120` 跑通，其余 19 个样本均为本样本 jobs 全部跑通。代表性失败日志报错为：

```text
Error in Stub::from_four_points():
Cannot create normalized xyzVector from vector of length() zero.
```

这提示 `6bk8` 至少存在一个几何退化/坐标构造问题，且失败明显偏向 `true_receptor`。此前观察到的 `pdb_UNK` 兼容性错误仍应保留为历史线索，但本次最终归因不能只写成 `pdb_UNK`，还需针对输入坐标和 Rosetta crash log 做专门排查。

assignment 审计共有 400 条记录，`solver` 字段均为 `dp_virtual`。这只说明本轮匹配使用虚拟节点 DP 路径，并不说明 assignment 是否选对。最终 summary 尚未包含 pose RMSD、RMSD <= 2/3/5 Å 比例、truth rank 或 top-k/top-k%；这些是下一步严格 evaluation 要补齐的结果。

### 时间预算口径修正与 realistic 尾部

`tables/runtime_budget_v1` 暂时保留为并行改造前 18 个完整样本的串行单核基线：`nstruct=5` 时成功 Rosetta job 的 P75 为 `331.1 s/job`，P90 为 `390.6 s/job`。不能将 `7ut7` 与 `7v19` 的新结果直接混进同一个分位表，因为二者分别使用 16 CPU 与 24 CPU 样本内并行；混合后单 job wall time 不再代表一种可复用的执行策略。

realistic 当前仍按每样本 1 CPU 串行运行。通过样本准备阶段已经生成的 complex 输入数和已生成 scorefile 数，可以直接得到尚未完成的 job 数与串行 ETA：

| 样本 | 计划 jobs | 已有 scorefiles | 剩余 jobs | P75 剩余时间 | P90 剩余时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8bly` | 364 | 36 | 328 | 30.2 小时 | 35.6 小时 |
| `8vcj` | 140 | 106 | 34 | 3.1 小时 | 3.7 小时 |
| `8ut3` | 1188 | 227 | 961 | 88.4 小时 | 104.3 小时 |
| `8wis` | 1680 | 292 | 1388 | 127.6 小时 | 150.6 小时 |
| `8ca3` | 2046 | 19 | 2027 | 186.4 小时 | 219.9 小时 |
| `6bk8` | 68 | 34 | 34 | 3.1 小时 | 3.7 小时 |
| `7z7s` | 1860 | 56 | 1804 | 165.9 小时 | 195.7 小时 |
| `8xh9` | 2160 | 161 | 1999 | 183.8 小时 | 216.9 小时 |

由于这些样本并行运行但各自串行处理内部 jobs，总完成时间由 `8ca3` 主导：在不改变运行方式的前提下，从本次检查起仍约需 `186` 小时（P75，约 7.8 天）到 `220` 小时（P90，约 9.2 天）。因此 realistic 的下一项调度动作应是评估如何将未完成尾部迁移到已经在 oracle 中验证有效的样本内并行执行方式，而不是静候串行 collector。
