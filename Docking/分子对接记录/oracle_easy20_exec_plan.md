# easy20 oracle / realistic docking ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

本计划负责本轮 easy20 统计、四类 oracle docking、以及前置预测较好样本 realistic docking。它从 `docking_master_exec_plan.md` 派生，所有本地修改限制在 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking`，服务器写入限制在 `/home/penghongen/分子对接尝试`。

## Purpose / Big Picture

本轮目标是先降低标准，判断当前 docking pipeline 的失败到底来自前置网络、ligand identity matching、还是 Rosetta 下游 refinement 本身。完成后应得到两套 easy20 样本集合和三批服务器输出：

1. `easy20_by_nk`：按 `n*k` 最小选出的 20 个计算量最小样本。
2. `easy20_by_instance_f1`：按前置网络 `instance_f1` 最高选出的 20 个前置预测最好样本。
3. `oracle_easy20`：在 `easy20_by_nk` 上运行四类 oracle 实验。
4. `realistic_easy20`：在 `easy20_by_instance_f1` 上运行真实预测 site docking。

最终用户可在 `oracle_easy20_experiment.md` 中看到每个配置项的语义、运行状态、结果和分析。

## Progress

- [x] (2026-05-21) 用户确认 easy20 的 `n` 是去纯金属离子后的 dockable ligand occurrence 数，`k` 是保守后处理后预测 site 数。
- [x] (2026-05-21) 用户确认保守后处理为不额外过滤小体素 instance，仅合并中心距离小于 8 Å 的原始 instance。
- [x] (2026-05-21) 用户确认 receptor 跑 `true_receptor` 与 `cryoatom_receptor`，服务器当前可用 96 CPU。
- [x] (2026-05-21) 用户确认 offset 半径 4/6/8 Å，每档 3 个 seed，默认 `nstruct=5`，小任务可后续提升到 10。
- [x] (2026-05-21) 用户确认四类 oracle 任务：`true_center_identity`、`offset_center_identity`、`true_center_hungarian`、`offset_center_hungarian`。
- [x] (2026-05-21) 新增 `Docking/run_easy20_prescan.py`，用于生成 easy20 统计表和样本列表。
- [x] (2026-05-21) 新增 `Docking/run_oracle_docking_batch.py`，用于按 task/variant 运行 oracle docking。
- [x] (2026-05-21) 新增 `easy20_prescan_cpu.sbatch`、`oracle_easy20_cpu.sbatch`、`realistic_easy20_cpu.sbatch`。
- [x] (2026-05-21) 新增 `oracle_easy20_experiment.md`，记录配置语义、输出位置、状态和后续结果分析。
- [x] (2026-05-21) 使用固定同步脚本同步到服务器，并只读确认服务器已有 `Docking/run_easy20_prescan.py` 与 `Docking/sbatch/easy20_prescan_cpu.sbatch`。
- [x] (2026-05-21) 提交 easy20 prescan CPU job：Slurm job `274525`，`PRESCAN_RUN_ID=20260521_easy20_prescan_v1`，8 CPU。
- [x] (2026-05-21) v1-v5 prescan 暴露并修复了三个问题：体素距离计算过重、重复扫描 ligand mapping、推理指标字段应读 `sample_name`。
- [x] (2026-05-21) 使用固定同步脚本同步 `sample_name` 修复，并提交 prescan v6：Slurm job `274539`，`PRESCAN_RUN_ID=20260521_easy20_prescan_v6`，8 CPU。
- [x] (2026-05-21) 读取 v6 prescan 输出并确认 `instance_f1` 已填入。v6 产出 80 个样本、60 个 usable 样本。
- [x] (2026-05-21) 根据 prescan v6 提交 oracle job `274540`：`RUN_ID=20260521_oracle_easy20_nk_nstruct5_v1`。
- [x] (2026-05-21) 提交 realistic job `274541`：`RUN_ID=20260521_realistic_easy20_instancef1_nstruct5_v1`，并设置 `afterany:274540` 依赖，避免超过 96 CPU。
- [x] (2026-05-21) 用户指出 96 核单 job 很难排队。已取消 Codex 提交且尚未运行的 `274540/274541`，改为 8 核 array 分片。
- [x] (2026-05-21) 为 `run_docking_batch.py` 和 `run_oracle_docking_batch.py` 增加 `--shard-id`，array task 只写 shard summary，避免并发覆盖总表。
- [x] (2026-05-21) 新增 `collect_batch_summary.py` 和 `collect_easy20_summary_cpu.sbatch`，用于 array 完成后汇总样本级 summary。
- [x] (2026-05-21) 提交 array 依赖链：oracle array `274549` -> oracle collect `274550` -> realistic array `274551` -> realistic collect `274552`。
- [x] (2026-05-21) 已完成只读 SSH 检查（10.102.33.220:10022）：oracle array `274549_0..11` 正在 RUNNING，`274549_12..19` 因 `JobArrayTaskLimit` 等待；`274550/274551/274552` 仍为 dependency 等待。oracle run 目录存在，但 `tables/batch_summary.json` 尚未生成（需等待 `274550` 汇总 job）。
- [x] (2026-05-21T18:11+08:00) 再次尝试本轮只读检查 `squeue/sacct` 与 oracle/realistic run 目录；但本机对 `10.102.33.220:10022` 的 SSH 在连接前即返回 `Permission denied`，且本地 `C:\Users\15919\.ssh` 目录不可读，因此本轮没有获得比 17:21+08:00 更新的远端事实。
- [x] (2026-05-21) SSH 连接已恢复并获得新反馈：oracle array `274549` 已完成 6/20 个样本，0 个失败 task；完成样本合计 240/240 个 Rosetta job 成功。`274550/274551/274552` 仍为 dependency pending。
- [x] (2026-05-22) 检查 oracle array 长尾：`274549` 已完成 12/20 个 array task，0 个 Slurm 失败 task；8 个 task 仍在运行，`274550/274551/274552` 仍为 dependency pending。
- [x] (2026-05-22) 定位到 `6bk8` 部分 Rosetta 单 job 失败：`RamaPrePro` 对 `pdb_UNK` residue type 无主链 score table，导致 returncode 1；这是 Rosetta job 级失败，不是 array/sbatch 失败。
- [ ] 运行完成后汇总流程跑通率、RMSD、assignment 和分层统计。

## Surprises & Discoveries

- Observation: 当前 runner 已经把 ligand PDB 的原子坐标均值平移到 site center。
  Evidence: `Docking/docking_pipeline/rosetta.py` 中 `translate_ligand_to_site()` 使用 `coords.mean(axis=0)` 计算 ligand 中心，再平移到 `site.center_world_xyz`。因此本轮 true center / offset center 实验与 Rosetta 输入语义相符。

- Observation: offset Hungarian 必须按半径和 seed 分成独立 variant，而不能把所有 offset site 混进一个 assignment。
  Evidence: 如果把一个真实 ligand 的 9 个偏移 site 同时放进一个样本级 assignment，会把“同一个真实位点的多个扰动版本”误当成多个不同预测位点。新 runner 使用 `task/variant` 目录隔离。

- Observation: 第一版 prescan 的中心距离合并参数触发了不必要的最近体素距离计算。
  Evidence: job `274525` 运行 5 分钟仍无表输出；原因是把 `merge_min_voxel_distance` 放宽到 `1e9` 仍会先计算 `_nearest_point_distance`。已取消该 Codex 提交的 job，并在 `instance_postprocess.py` 中加入中心距离模式：当最近体素距离阈值和 bbox 增量阈值都大于等于 `1e8` 时，跳过点云距离和 bbox 计算。

- Observation: 第二版 prescan 仍然不够轻，因为即使跳过距离计算，`postprocess_sites` 仍会为每个 instance 准备完整体素坐标。
  Evidence: job `274528` 运行 3 分钟仍无表输出。已取消该 Codex 提交的 job，并把 `run_easy20_prescan.py` 改为只读取 `voxel_candidates.json` 中的 center，使用 center-only union-find 统计保守合并后的 `k`，不再读取 `instance_label`。

- Observation: center-only prescan v3 已经不慢，但有一个直接导入错误。
  Evidence: job `274531` 2 秒失败，错误为 `NameError: name 'np' is not defined`。已补充 `import numpy as np`，下一步提交 v4。

- Observation: v4 center-only prescan 仍然超过预期，瓶颈可能是每个样本重复扫描 ligand mapping CSV。
  Evidence: job `274532` 运行 3 分钟仍无表输出，已取消。已把 `run_easy20_prescan.py` 改为一次性读取 `ligand_mapping.csv` 并按样本缓存候选 ligand。

- Observation: v5 prescan 成功，但 `instance_f1` 全为空，不能作为前置预测最好样本集合。
  Evidence: `20260521_easy20_prescan_v5` 1 分 14 秒完成，`manifest.json` 有 80 个样本、60 个 usable；但 `easy20_by_instance_f1.csv` 中 `instance_f1` 为空。服务器检查发现 `per_sample_best_metrics.json` 使用字段 `sample_name`，已修正指标归一化。

- Observation: 用户新增的第四类 `offset_center_hungarian` 已纳入默认 oracle 任务。
  Evidence: `Docking/run_oracle_docking_batch.py` 的 `TASKS` 和 `Docking/sbatch/oracle_easy20_cpu.sbatch` 的 `TASKS` 默认值都包含 `true_center_identity`、`offset_center_identity`、`true_center_hungarian`、`offset_center_hungarian`。

- Observation: prescan v6 的 `easy20_by_instance_f1` 不再退化为 `nk` 排序，但前置预测好不等于 docking 简单。
  Evidence: v6 的 `easy20_by_instance_f1` 包含 `8ut3`、`8wis`、`8xh9` 等 ligand 数或糖类比例很高的样本；后续 realistic 需要按 ligand 数、糖类比例、内部金属、large ligand 与前置指标分层。

- Observation: 96 核单 job 在当前 CPU 队列上排队困难，8 核 array 更符合调度现实。
  Evidence: `274540` 单 job 申请 96 核后停在 `(Resources)`；改为 `274549` array 后，`274549_0` 到 `274549_11` 立即运行，剩余 task 因 `%12` array limit 正常等待。当前最大并发为 12 task * 8 核 = 96 核。
- Observation: 本轮新的阻塞不是 array 自身报错，而是当前工作站到服务器 SSH 入口重新变得不可达，因此不能把旧的 17:21+08:00 观测误写成最新状态。
  Evidence: `ssh -o BatchMode=yes -o ConnectTimeout=10 -p 10022 penghongen@10.102.33.220 "echo ok"` 直接返回 `Permission denied`；`Get-ChildItem`/`Test-Path` 访问 `C:\Users\15919\.ssh` 也返回 `Access is denied`。

- Observation: 上一条 SSH 阻塞已被后续成功连接覆盖；当前应以 oracle array 的新状态为准。
  Evidence: 本轮成功读取 `squeue`、`sacct` 和 `/home/penghongen/分子对接尝试/pipeline_runs/20260521_oracle_easy20_nk_nstruct5_array8_v1`。已完成样本级 summary 共 6 个：`8dd7`、`8x9s`、`7vla`、`8v7l`、`8p71`、`8wpf`，全部 `status=ok`。

- Observation: oracle array 的前 6 个完成样本流程跑通率为 100%，暂未出现 Rosetta stderr。
  Evidence: 完成样本合计 `num_variants=120`、`num_jobs=240`、`num_success=240`。`slurm_oracle_easy20_274549_*.err` 暂无非空典型错误日志。

- Observation: 当前 oracle 运行的长尾来自样本计算量差异，而不是排队或脚本错误。
  Evidence: 已完成的 array task 为 `274549_0/1/2/3/6/7`；仍运行的 task 对应 `7zdf`、`8v6v`、`8pmd`、`7n70`、`7nnl`、`7tju`、`7zdl`、`8umt`、`8wox`、`8vm0`、`6bk8`、`7ut7`；`8x0b`、`7v19` 因 `%12` array limit 尚未启动。

- Observation: 到 2026-05-22 检查时，oracle array 已从“排队长尾”转为“计算长尾”，且已经完成 12/20 个样本。
  Evidence: `sacct` 显示 `274549_0/1/2/3/4/5/6/7/8/9/12/14` 为 `COMPLETED` 且 `ExitCode=0:0`；`274549_10/11/13/15/16/17/18/19` 仍在运行。样本级 summary 显示已完成样本合计 240 个 variant、1120 个 Rosetta job、1120 个成功 Rosetta job。

- Observation: `6bk8` 是当前第一个明确出现 Rosetta job 级失败的样本，失败集中在 `IHP`/`GTP` 相关 true/offset oracle variant，尤其 true receptor 路径。
  Evidence: `6bk8/variants/*/*/audit/summary.json` 中多处 `num_success < num_jobs`。失败日志 `dock_site001_true_receptor_IHP.stderr.log` 与 `ROSETTA_CRASH.log` 报错：`RamaPrePro::get_mainchain_torsions_covered(): No mainchain score table for residue type pdb_UNK exists`。这说明 sbatch array 仍在正常运行，但 Rosetta 对输入中某些 `UNK` residue type 不能做 RamaPrePro 主链项。

## Decision Log

- Decision: easy20 统计使用 `min_voxels=1`、`merge_center_distance=8.0`、其他合并约束放宽到极大值。
  Rationale: 用户要求本轮保守一些，不额外去除小体素 instance，并尽量让合并只由中心距离控制。
  Date/Author: 2026-05-21 / 用户与 Codex

- Decision: oracle identity 与 oracle Hungarian 共用同一套 true/offset centers，但 identity 只跑对应真实 ligand，Hungarian 跑全部候选 ligand。
  Rationale: 这样可以正交区分“中心是否正确”和“ligand identity 是否需要靠打分判断”。
  Date/Author: 2026-05-21 / 用户与 Codex

- Decision: realistic easy20 不写新 runner，继续调用 `run_docking_batch.py`。
  Rationale: realistic 实验应该代表真实 pipeline，只通过 CLI 参数指定保守后处理，避免 oracle 逻辑污染真实流程。
  Date/Author: 2026-05-21 / Codex

- Decision: 长 CPU 任务优先使用 8 核 array 分片，而不是 96 核单 job。
  Rationale: 8 核 task 更容易被 Slurm 填入空闲资源；使用 array `%12` 可以保持总 CPU 不超过 96，同时不要求整台 96 核机器同时空闲。
  Date/Author: 2026-05-21 / 用户与 Codex

## Outcomes & Retrospective

当前处于本地实现完成、等待同步和服务器运行阶段。本地已通过 `Pocket_Plus_windows` 环境下两个新脚本的 `--help` 检查；还需要服务器侧 prescan 真正验证推理输出文件名和逐样本 `instance_f1` 读取口径。

2026-05-21 更新：固定同步脚本已把新增脚本同步到服务器；`ssh` 只读确认服务器文件存在。已提交 prescan job `274525`，但运行 5 分钟仍无表输出，判断为中心距离合并实现触发了不必要的最近体素距离计算。该 job 由 Codex 提交，已取消。随后提交修正版 job `274528`，但 3 分钟仍无表输出，进一步判断瓶颈是读取完整 `instance_label` 并准备每个 instance 的体素坐标。已取消 `274528`，并把 prescan 改为 center-only union-find。提交 v3 job `274531` 后 2 秒失败，原因是漏导入 numpy；已修复。提交 v4 job `274532` 后 3 分钟仍无表输出，判断重复扫描 ligand mapping CSV 仍太慢；已取消并改为一次性缓存 mapping。v5 job `274535` 成功，但 `instance_f1` 为空；已修正 `sample_name` 字段读取。修复已同步到服务器，并提交 v6 job `274539`。v6 8 秒完成，`instance_f1` 已正确填入。最初提交的 96 核单 job `274540/274541` 因排队困难已取消；现已改为 array 依赖链 `274549 -> 274550 -> 274551 -> 274552`。

2026-05-21T18:11+08:00 更新：本轮优先尝试按自动化要求重新检查 `squeue/sacct`、oracle run 审计计数与 realistic 是否启动。但当前工作站对 `10.102.33.220:10022` 的 SSH 连接在建立前即返回 `Permission denied`，而本地 `C:\Users\15919\.ssh` 目录也因权限限制不可读，无法回退到 key 配置检查。因此本轮没有新的服务器事实；最后可信远端观察仍是 17:21+08:00：oracle array `274549` 运行中、`274550/274551/274552` 仍在 dependency 等待、variant 级 `audit/summary.json` 已有 49 个、`tables/batch_summary.json` 尚未生成。

2026-05-21 后续更新：SSH 已恢复；上一条 Permission denied 不是当前阻塞。oracle array `274549` 仍在运行，已完成 6 个样本：`8dd7`、`8x9s`、`7vla`、`8v7l`、`8p71`、`8wpf`。这些样本全部 `status=ok`，合计 240 个 Rosetta job 全部成功。当前仍无 oracle `batch_summary.json`，因为汇总 job `274550` 仍在 dependency pending；realistic `274551` 尚未开始。

2026-05-22 更新：oracle array `274549` 仍在运行；已完成 12/20 个 array task，0 个 Slurm 失败。完成样本为 `8dd7`、`8x9s`、`7vla`、`8v7l`、`7zdf`、`8v6v`、`8p71`、`8wpf`、`8pmd`、`7n70`、`7zdl`、`8wox`，样本级 summary 合计 `num_variants=240`、`num_jobs=1120`、`num_success=1120`。未完成但正在运行的样本为 `7nnl`、`7tju`、`8umt`、`8vm0`、`6bk8`、`7ut7`、`8x0b`、`7v19`。`6bk8` 的 partial variant summary 已显示 Rosetta job 级失败，典型错误为 `pdb_UNK` 缺少 RamaPrePro 主链 score table。realistic 仍未开始。

## Context and Orientation

相关入口：

- `Docking/run_easy20_prescan.py`
- `Docking/run_oracle_docking_batch.py`
- `Docking/run_docking_batch.py`
- `Docking/sbatch/easy20_prescan_cpu.sbatch`
- `Docking/sbatch/oracle_easy20_cpu.sbatch`
- `Docking/sbatch/realistic_easy20_cpu.sbatch`
- `Docking/分子对接记录/oracle_easy20_experiment.md`

预期服务器输出：

```text
/home/penghongen/分子对接尝试/easy20_prescans/20260521_easy20_prescan_v6
/home/penghongen/分子对接尝试/pipeline_runs/20260521_oracle_easy20_nk_nstruct5_array8_v1
/home/penghongen/分子对接尝试/pipeline_runs/20260521_realistic_easy20_instancef1_nstruct5_array8_v1
```

## Plan of Work

第一步，用固定同步脚本把本地 `Docking` 更新到服务器。

第二步，提交 `easy20_prescan_cpu.sbatch`。它只做统计，不运行 Rosetta，应该很快完成。完成后读取 `manifest.json`、`easy20_by_nk.csv`、`easy20_by_instance_f1.csv`，检查样本数量、`instance_f1` 是否读到、`nk` 是否合理、糖类/大 ligand/内部金属比例是否有异常。

第三步，若 prescan 表合理，提交 `oracle_easy20_cpu.sbatch` 和 `realistic_easy20_cpu.sbatch`。两者总 CPU 不能超过 96；如果同时跑，应各自降低 `JOBS` 或排队提交。若资源不足，优先跑 oracle，因为它直接回答下游上限问题。

2026-05-21 修订：CPU 任务不再使用 96 核单 job。`oracle_easy20_cpu.sbatch` 和 `realistic_easy20_cpu.sbatch` 已改为 `#SBATCH --cpus-per-task=8` 与 `#SBATCH --array=0-19%12`。每个 array task 跑一个样本，`--shard-id` 写入独立 shard summary，最终由 `collect_easy20_summary_cpu.sbatch` 汇总。

第四步，等待任务完成后运行 evaluation。oracle 输出采用 task/variant 目录结构，可能需要补一个专门的 evaluation reader；realistic 输出沿用现有 `run_evaluation.py`。

第五步，把流程跑通、RMSD、assignment、按 ligand 类型分层的结果写回 `oracle_easy20_experiment.md`，同时更新本 ExecPlan 的 Progress、Surprises & Discoveries 和 Outcomes & Retrospective。

## Validation and Acceptance

本地验收：

- `python -m compileall Docking` 通过。
- `Pocket_Plus_windows` 环境下 `run_easy20_prescan.py --help` 通过。
- `Pocket_Plus_windows` 环境下 `run_oracle_docking_batch.py --help` 通过。

服务器验收：

- prescan 输出包含 `all_samples.csv`、`easy20_by_nk.csv`、`easy20_by_instance_f1.csv`、两个样本列表和 `manifest.json`。
- oracle run 的每个样本包含 `variants/{task}/{variant_id}/audit/summary.json`。
- realistic run 沿用 `samples/{sample_id}/audit/summary.json`。
- 所有输出只写入 `/home/penghongen/分子对接尝试`。

## Idempotence and Recovery

所有 run id 固定后，重复提交会覆盖或追加同一 run 目录下自己生成的审计文件，不写出允许目录。若 prescan 发现 `instance_f1` 为空，应先修正读取口径，不直接提交真实 Rosetta 任务。若 oracle 或 realistic 任务运行时间过长，通过 Slurm job id 检查状态；只允许取消 Codex 本轮提交的 job。
