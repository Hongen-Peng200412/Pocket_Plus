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
- [x] (2026-05-23) 确认当前 8 核 array task 实际基本为单活跃 CPU 核运行：`JOBS=1` 且线程变量均为 1，Slurm `TotalCPU` 接近 wall time 而非 8 倍 wall time。这是严重的资源申请性能 bug。
- [x] (2026-05-23) 新增 `Docking/analyze_runtime_budget.py`，从每个 `audit/results.json` 的 `seconds` 字段汇总单 job 实测耗时，并在服务器 oracle run 下生成 `tables/runtime_budget_v1` 报告。
- [x] (2026-05-23) 运行时间基线刷新为 18 个完整样本：成功 job `3459` 个，失败 job `61` 个；成功 job 的 Median/P75/P90 分别为 `220.3/331.1/390.6` 秒。以后提交新任务默认按 P75 预算，并同时报告长尾 P90 上界。
- [x] (2026-05-23) 核实 oracle 实验的成本由真实 ligand 数 `n` 决定：当前四类任务总 job 数为 `20*n*(n+1)`，而 prescan `nk` 不是 oracle 的直接成本公式。`7v19` 的 `n=8`、完整成本 `1440` jobs；`7ut7/8x0b` 的 `n=6`、完整成本 `840` jobs。
- [x] (2026-05-23) 根据用户认可的临时快速验证原则，取消 Codex 提交的长尾 array task `274549_19`（`7v19`），保留 partial 输出供诊断；本轮 oracle 完整执行集定义为 `easy19_fast_validation`。`8x0b` 已完成，只剩 `7ut7` 运行。
- [x] (2026-05-23) 修复下一轮默认 sbatch 资源申请：oracle/realistic 每个串行 array task 改为 `1 CPU`，array 并发上限为 `%96`；汇总任务改为 `1 CPU`。本修改不影响已在运行的旧任务。
- [x] (2026-05-23) 将性能修复立即用于 realistic：取消低效旧 run 的 Codex jobs `280768/280769`，提交新 run `20260523_realistic_easy20_instancef1_nstruct5_array1_fixed_v1`，array job `281985` 的 20 个 1-CPU task 已同时运行，汇总 job 为 `281986`。与旧 oracle `7ut7` 并行时总申请 CPU 为 28，低于 96 上限。
- [x] (2026-05-23T21:03+08:00) Heartbeat 检查新 run：realistic `281985` 已完成 5/20 个样本（`8dd7, 7zdf, 8pmd, 8v6v, 7zdl`），样本级 summary 合计 `60/60` 个 Rosetta jobs 跑通；oracle 仍仅有 `7ut7` 运行，最后可审计的剩余预算仍为 P75 `19.9` 小时 / P90 `23.4` 小时。
- [x] (2026-05-23) 根据用户决定实现通用的样本内部并行层：`docking_pipeline.rosetta.run_rosetta_jobs(..., rosetta_jobs)` 对整个样本的独立 Rosetta job 队列并发执行，`run_docking_batch.py` 和 `run_oracle_docking_batch.py` 均新增 `--rosetta-jobs` 入口。
- [x] (2026-05-23) oracle runner 已从“逐 variant 串行运行”改为“先准备全部 variant 的 jobs，再把所有 ligand-site-receptor jobs 放进同一并行池，最后按原顺序切回 variant 做 assignment 与审计”；因此 identity/Hungarian、true/offset 产生的 jobs 都受同一并行能力覆盖。
- [x] (2026-05-23) 为避免并发失败时多个 Rosetta 子进程覆盖共享的 `ROSETTA_CRASH.log`，将每个子进程的工作目录改为该 job 独占的 `output_dir`；并新增 `sbatch/oracle_single_sample_parallel_cpu.sbatch` 供单样本全量覆盖与验证。
- [x] (2026-05-23T21:47+08:00) 首次提交 `7v19` 24 核全量覆盖验证 job `283182`，但它在 25 秒内得到样本级 `CalledProcessError`：旧 partial 输出已含 `true_center_identity/center/params/L01.params` 等文件，再次执行 `molfile_to_params.py` 失败，尚未进入 Rosetta 并行执行。
- [x] (2026-05-23) 修复批处理失败状态传播：`run_oracle_docking_batch.py` 与 `run_docking_batch.py` 在写入样本失败摘要后返回非零退出码，避免类似 `283182` 被 Slurm 错标为 `COMPLETED`。
- [x] (2026-05-23T21:52+08:00) 仅删除已获授权覆盖的 `samples/7v19` 旧 partial 输出后，重新提交 `24 CPU` 全量验证 job `283183`；首次状态检查显示其已在 `cnode01` 运行并重新生成 899 个准备文件，已越过参数文件冲突。
- [x] (2026-05-23T21:53+08:00) `283183` 的首轮运行证据已证明多核实际工作：wall time `1:37` 时 `sstat` 报告累计 CPU 时间 `17:28`，约为墙钟的 `10.8x`，明显不同于旧串行任务约 `1x` 的行为。
- [x] (2026-05-23T21:56+08:00) 第二轮监视中，`283183` 在 wall time `6:26` 已累计 CPU `2:12:50`（约 `20.6x`），stderr 与样本 error 仍为空；`cnode01` 约有 2 TB 内存，约 1.87 TB 空闲，当前并行内存消耗不是阻塞。
- [x] (2026-05-23) 基于真实并行证据，取消 Codex-owned 旧串行 task `274549_17`（`7ut7`）与会过早汇总 partial 输出的旧 collector `274550`；仅清空获授权覆盖的 `samples/7ut7` 后提交 16 核全量重跑 job `283281`。
- [x] (2026-05-23) 提交最终 oracle 汇总 job `283282`，依赖 `afterany:283183:283281`；本轮最终分析目标恢复为完整 `easy20`，而 `easy19_fast_validation` 仅保留为并行修复前的临时决策记录。
- [x] (2026-05-23T22:02+08:00) 落盘验证完成：`283183` 在 wall `9:32` 时累计 CPU `3:27:10`（约 `21.7x`），已有 24 个 scorefile / 76 个 output 文件且 stderr 为空；`283281` 在 wall `1:51` 时累计 CPU `17:32`（约 `9.5x`），已开始使用多核。
- [x] (2026-05-24 12:02+08:00) `283183/283281/283282` 全部完成；oracle 最终集合为完整 `easy20`，最终 summary 覆盖 20 个样本、400 个 variants、5800 个 Rosetta jobs。
- [x] (2026-05-24 12:02+08:00) oracle 流程跑通汇总完成：5739/5800 jobs 流程跑通（98.95%）；61 个失败全部来自 `6bk8`，代表性 stderr 为 Rosetta `Stub::from_four_points()` zero-length vector 内部错误。
- [x] (2026-05-24 12:02+08:00) oracle assignment 审计完成：总计 400 条 assignment，solver 字段均为 `dp_virtual`；汇总产物当前未包含 RMSD/assignment correctness，不能报告真正 docking 成功率。
- [x] (2026-05-24 12:02+08:00) realistic `281985` 已有 12/20 样本 summary 且 636/636 jobs 流程跑通；8 个未完成串行样本中，`8ca3` 按未完成 job 数与串行 P75/P90 仍约需 186/220 小时。
- [x] (2026-05-24 13:18+08:00) 实现 `evaluation/run_oracle_evaluation.py` 与 `sbatch/oracle_evaluation_cpu.sbatch`，按真实 site occurrence 评价 assignment、truth-pair pose RMSD、严格选中 pose RMSD 与 top-k/top-k%。
- [x] (2026-05-24 13:25+08:00) 修复前 oracle 的即时严格评价 job `284650` 已完成：严格选中 RMSD `<=2/3/5 Å` 为 `0.22%/2.99%/14.24%`，Hungarian occurrence accuracy 为 `50.18%`，真实 pair top-1 为 `42.95%`。
- [x] (2026-05-24 13:25+08:00) `6bk8` 归因为 true receptor 引入 `UNK` polymer residue；`io_utils.cif_to_receptor_pdb()` 已过滤该 Rosetta 不支持残基并维持断链，修复 smoke `284651` 的 true-receptor identity jobs 为 `2/2` 跑通。
- [x] (2026-05-24 13:26+08:00) 获授权删除原 oracle run 中仅 `samples/6bk8` 的旧结果，提交完整覆盖 `284663`、collector `284664` 和最终严格评价 `284665`。
- [x] (2026-05-24 13:24+08:00) 获授权迁移 realistic 尾部：保留已完成的 `6bk8` 与仍接近完成的 `8vcj`，覆盖六个长尾样本并提交 jobs `284652..284657` 与 collector `284658`。
- [x] (2026-05-24 13:27+08:00) heartbeat 复查确认 `284663` 和 `284652..284657` 均处于 RUNNING、stderr 为空；oracle 覆盖任务在 `00:06:38` wall time 已累计 `00:56:12` CPU，realistic 当前已有 `13/20` 个样本 summary。
- [ ] 读取 `284665` 最终 oracle 质量表并写入 `oracle_easy20_readable.md`；读取 `284658` 后运行 realistic 严格评价与分层统计。

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
- Observation: 当前 array 资源配置浪费约 7/8 的调度 CPU 配额，不能继续作为后续实验默认设置。
  Evidence: `oracle_easy20_cpu.sbatch` 申请 `--cpus-per-task=8`，同时设置 `JOBS=1` 与 `OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS=1`；已完成任务例如 `8dd7` 的 Slurm wall time 为 `6971 s`、`TotalCPU=6931 s`，而非接近 `8*6971 s`。
- Observation: 每个 Rosetta 子 job 已有直接耗时记录，不必用样本 wall time 倒推。
  Evidence: `docking_pipeline/rosetta.py::run_rosetta_job()` 用 `time.time()` 记录 `seconds`，并由 oracle runner 写入各 variant 的 `audit/results.json`。
- Observation: `7v19` 是 oracle 设计中的自然重型样本，而不只是 prescan `nk` 口径下的偶然慢点。
  Evidence: prescan 中 `7v19` 的 `nk=48`、真实 dockable ligand 数 `n=8`；`7ut7` 与 `8x0b` 均为 `nk=36`、`n=6`。oracle 用真实中心和真实 ligand 候选，完整 job 数为 `20*n*(n+1)`，因此 `7v19=1440` jobs，后两者各 `840` jobs。
- Observation: 1-CPU realistic 重提已获得首批流程反馈，说明资源修复没有阻止 runner 正常落盘。
  Evidence: `281985` 运行约 2.6 小时后已有 5 个样本级 summary，`8dd7/7zdf/8pmd/8v6v/7zdl` 合计 `num_jobs=60` 且 `num_success=60`；这是流程跑通证据，不是 assignment 或 RMSD 科学结论。
- Observation: 仅把 array task 的 CPU 申请数改大不能自动加速一个样本，必须让 runner 把独立 Rosetta jobs 真正调度到并行执行器。
  Evidence: 旧任务申请 8 CPU 仍只有约 1 CPU 活跃；新实现把 realistic 样本的全部 site-ligand-receptor jobs 与 oracle 样本跨 variant 的全部 jobs 统一交给 `run_rosetta_jobs(..., rosetta_jobs)`。
- Observation: 样本内部并发还带来一个原本串行时不明显的输出冲突风险。
  Evidence: 旧 `run_rosetta_job()` 以 variant 目录作为多个 job 的公共工作目录；Rosetta 失败时可能在该目录竞争写 `ROSETTA_CRASH.log`。新实现让每个子进程在独占 `output_dir` 下运行，日志和 decoy 仍按 job 隔离。
- Observation: 原 oracle run 的 partial 样本目录不能直接被完整重跑复用，至少 `molfile_to_params.py` 的生成步骤不是覆盖幂等的。
  Evidence: `283182` 在 25 秒结束，`samples/7v19/audit/error.json` 显示其在 `true_center_identity/center/inputs/ligands/CLR_01.mol2` 参数生成阶段退出；同一 variant 的 `params/` 中已经存在 `L01.params`、`L01_0001.pdb` 等旧 partial 文件。由于用户已授权只覆盖 `7v19`，安全恢复方式是删除该单一样本输出后全量重跑。
- Observation: 批处理入口原先会把样本失败写进 JSON，却仍以进程退出码 0 结束，导致 Slurm 状态无法代表样本准备/执行是否成功。
  Evidence: `283182` 的 stdout 明确写出 `num_failed=1` 与 `CalledProcessError`，但 `sacct` 为 `COMPLETED ExitCode=0:0`；已在两个 batch 入口中加入失败摘要落盘后的非零退出。
- Observation: `--rosetta-jobs` 已在真实长尾样本上产生多核执行，而不是再次出现空申请。
  Evidence: 重提的 `7v19` job `283183` 分配 24 CPU；运行 `00:09:32` 时，`sstat -j 283183.batch` 报告累计 CPU `03:27:10`，累计 CPU 与墙钟比约 `21.7`，且已有 24 个 scorefile / 76 个 output 文件写出，stderr 为 0 字节。新 `7ut7` job `283281` 分配 16 CPU；运行 `00:01:51` 时累计 CPU `00:17:32`，约 `9.5x`。
- Observation: 失败验证 shard `283182` 会保留为执行历史，但不会污染最终 easy20 的样本级统计。
  Evidence: `collect_batch_summary.py` 以 `samples/*/audit/summary.json` 构造 `num_samples/num_jobs/num_success/samples`，仅以 `tables/shards/*_summary.json` 的数量填充 `num_shards` 审计字段；全量覆盖写回唯一的 `samples/7v19/audit/summary.json` 后不会重复计入科学汇总。

- Observation: 完整 easy20 oracle 已经形成可分析总表，长尾覆盖任务本身没有引入新的 Rosetta 失败。
  Evidence: `283183`（`7v19`）以 24 CPU 完成于 `10:29:41` 且为 `1440/1440` jobs 跑通；`283281`（`7ut7`）以 16 CPU 完成于 `04:50:59` 且为 `840/840` jobs 跑通；最终总表的 61 个失败全部归于 `6bk8`。

- Observation: 当前 oracle 的失败具有明显 receptor 偏置，并且错误类型与此前记录的 `pdb_UNK` 现象不同或至少不完整等价。
  Evidence: `true_receptor` 为 `2840/2900` 跑通，而 `cryoatom_receptor` 为 `2899/2900` 跑通；本次直接读取的代表性失败日志报 `Stub::from_four_points()` 与 zero-length normalized vector。后续排错需同时核对 `ROSETTA_CRASH.log` 和输入坐标，而不能仅沿用早期 `pdb_UNK` 归因。

- Observation: realistic 旧修复策略解决了“8 核空申请”但尚未利用已验证的样本内并行能力。
  Evidence: `281985` 每 task 为 1 CPU 串行，已完成 12 个样本；未完成的 `8ca3`、`8xh9`、`7z7s` 分别还有约 `2027/1999/1804` 个已准备但未产生 scorefile 的 jobs，按 P75/P90 仍是数日尾部。

- Observation: `6bk8` true receptor 的主要失败根因是 receptor-only PDB 转换未排除 Rosetta 无法处理的 `UNK` polymer residue。
  Evidence: 旧 true receptor 含 `1175` 个 `UNK` ATOM，cryoatom receptor 无 `UNK`；失败日志出现 `RamaPrePro... residue type pdb_UNK`，且过滤后 `20260524_oracle_6bk8_unk_filter_smoke_v1` 的 `2/2` true receptor identity jobs 成功。

- Observation: 专用 oracle evaluator 揭示出“中心 oracle”下 pose 上限仍低，而匹配错误进一步损失结果。
  Evidence: 修复前评价 `20260524_oracle_easy20_strict_eval_prefixed_v1` 中，`truth_pair_pose_summary` 的可靠 RMSD `<=5 Å` 为 `449/2200 = 20.41%`，而 `strict_selected_pose_summary` 为 `319/2240 = 14.24%`；Hungarian 匹配 `562/1120 = 50.18%`。

- Observation: `6bk8` 权威覆盖运行阶段已经复现多核加速特征，且目前未复现旧 `UNK` stderr。
  Evidence: `284663` 申请 24 CPU；在墙钟 `00:06:38` 时 `sstat` 累计 CPU 为 `00:56:12`，stderr 仍为 0 字节。`284664/284665` 仍按依赖等待，因此最终评价尚未形成。

- Observation: realistic 覆盖迁移保持健康，已完成 summary 没有因替换长尾而被丢弃。
  Evidence: 服务器当前可见 13 个 realistic 样本级 `audit/summary.json`；六个覆盖 jobs `284652..284657` 均在运行且 stderr 为空，保留的原 `8vcj` task 仍运行，collector `284658` 等待全部依赖。

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

- Decision: 对未启用样本内部并行的历史任务，串行 Rosetta array task 只应申请 1 CPU；该规则已被后续“显式启用 `--rosetta-jobs` 的多核样本任务”扩展，而不是继续作为唯一主线。
  Rationale: 已验证旧 `JOBS=1` 运行只活跃使用约 1 核；新 runner 则显式启动多个 Rosetta 子进程，可以合理申请与 `rosetta_jobs` 相匹配的多核资源。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: 运行时间预算先以成功单 job 的 P75 秒数作为默认值，同时报告 P90 长尾上界；失败 job 只计入实际消耗账本，不混入正常预算模型。
  Rationale: AI agent 需要看到 job 数即可立即估算成本，且不能被 `6bk8` 的 Rosetta 输入失败扭曲正常耗时。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: 本轮 oracle 快速验证集改称 `easy19_fast_validation`，取消 `7v19` 的未完成长尾任务，不用候补样本补满 20。
  Rationale: easy 集合是临时快速实验面板；`7v19` 的真实 ligand 数使 oracle 完整成本显著更高，继续等待将延迟数日而不改变当前快速验证目的。原始 easy20 选择与 `7v19` partial 输出保留用于审计和困难样本诊断。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: 已启动但仍处于早期阶段的 realistic 旧 run 不继续承受 8 核申请性能 bug，直接以新 run ID 重新提交 1-CPU array。
  Rationale: 旧 run 最多只推进 9 条单核串行 task 却申请 72 CPU；新 run 只申请 20 CPU 即可同时推进全部 20 个样本。保留旧 run 作为 superseded partial，分析只读取新 run，避免混合输出。
  Date/Author: 2026-05-23 / Codex

- Decision: 将样本内部并行实现为与 job 来源无关的公共执行层，并用于 oracle 与后续 realistic 任务。
  Rationale: identity 与 Hungarian 的本质区别在于是否随后需要 assignment，而不是单个 Rosetta 子进程如何运行；把所有 ligand-site-receptor jobs 统一排入样本级执行池，既可复用，也能让重样本真正利用多核。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: 分配 CPU 的推荐经验规则为 `ceil(计划 Rosetta jobs / 40)`，资源宽裕或希望更积极缩短长尾时可参考 `ceil(jobs / 30)`；这只是建议，不是代码硬限制。
  Rationale: 规则足够简单，未来 agent 看见样本预计 job 数就能立刻提出可解释的资源预算，同时允许按队列可用性、任务优先级和总计不超过 96 核的边界调整。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: 先在原 oracle run 中对 `7v19` 做 24 核全量覆盖验证；若有效，将此前暂时的 `easy19_fast_validation` 决策撤回并恢复 easy20 分析，同时仅在旧 `7ut7` 尚未完成时取消其 Codex-owned 旧 task 后做 16 核全量覆盖。
  Rationale: 用户允许覆盖仅涉及该样本的 partial 输出；`7v19` 全量 1440 jobs 恰好能验证真正长尾是否被并行消解，而不用为验证另建一套难以合并的结果目录。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: `283183` 的实际 CPU 证据足以将样本内部并行提升为当前 oracle 主线，并用 `283281` 覆盖旧 `7ut7`；原 collector `274550` 取消，改由 `283282` 在两项覆盖结束后生成最终 easy20 汇总。
  Rationale: `7v19` 在 6 分钟观察窗口内达到约 `20.6x` CPU/墙钟比、无 stderr 且内存余量充分；继续让 `7ut7` 旧串行尾部阻塞汇总不再合理。取消旧 collector 可防止覆盖重跑期间生成误导性的缺样本总表。
  Date/Author: 2026-05-23 / Codex

- Decision: 保留 `runtime_budget_v1` 作为 18 个完整串行样本的单核时间预算基线，不用 16/24 CPU 覆盖后的 `7ut7/7v19` 结果无标注地重算同一分位表。
  Rationale: 并行执行下的单 job wall time受到同节点并发竞争影响；未来需要单独记录并行策略的样本级 wall time/核数/加速比，才能让 ETA 可复用。
  Date/Author: 2026-05-24 / Codex

- Decision: `6bk8` 修复先以独立 smoke run 验证，再覆盖 authoritative oracle sample；最终评价依赖覆盖后的 collector。
  Rationale: 该顺序既验证修复是否触及主要根因，又不会在修复未经证明时破坏唯一权威样本输出。用户已经授权在 smoke 成立后进行覆盖。
  Date/Author: 2026-05-24 / 用户与 Codex

## Outcomes & Retrospective

当前处于本地实现完成、等待同步和服务器运行阶段。本地已通过 `Pocket_Plus_windows` 环境下两个新脚本的 `--help` 检查；还需要服务器侧 prescan 真正验证推理输出文件名和逐样本 `instance_f1` 读取口径。

2026-05-21 更新：固定同步脚本已把新增脚本同步到服务器；`ssh` 只读确认服务器文件存在。已提交 prescan job `274525`，但运行 5 分钟仍无表输出，判断为中心距离合并实现触发了不必要的最近体素距离计算。该 job 由 Codex 提交，已取消。随后提交修正版 job `274528`，但 3 分钟仍无表输出，进一步判断瓶颈是读取完整 `instance_label` 并准备每个 instance 的体素坐标。已取消 `274528`，并把 prescan 改为 center-only union-find。提交 v3 job `274531` 后 2 秒失败，原因是漏导入 numpy；已修复。提交 v4 job `274532` 后 3 分钟仍无表输出，判断重复扫描 ligand mapping CSV 仍太慢；已取消并改为一次性缓存 mapping。v5 job `274535` 成功，但 `instance_f1` 为空；已修正 `sample_name` 字段读取。修复已同步到服务器，并提交 v6 job `274539`。v6 8 秒完成，`instance_f1` 已正确填入。最初提交的 96 核单 job `274540/274541` 因排队困难已取消；现已改为 array 依赖链 `274549 -> 274550 -> 274551 -> 274552`。

2026-05-21T18:11+08:00 更新：本轮优先尝试按自动化要求重新检查 `squeue/sacct`、oracle run 审计计数与 realistic 是否启动。但当前工作站对 `10.102.33.220:10022` 的 SSH 连接在建立前即返回 `Permission denied`，而本地 `C:\Users\15919\.ssh` 目录也因权限限制不可读，无法回退到 key 配置检查。因此本轮没有新的服务器事实；最后可信远端观察仍是 17:21+08:00：oracle array `274549` 运行中、`274550/274551/274552` 仍在 dependency 等待、variant 级 `audit/summary.json` 已有 49 个、`tables/batch_summary.json` 尚未生成。

2026-05-21 后续更新：SSH 已恢复；上一条 Permission denied 不是当前阻塞。oracle array `274549` 仍在运行，已完成 6 个样本：`8dd7`、`8x9s`、`7vla`、`8v7l`、`8p71`、`8wpf`。这些样本全部 `status=ok`，合计 240 个 Rosetta job 全部成功。当前仍无 oracle `batch_summary.json`，因为汇总 job `274550` 仍在 dependency pending；realistic `274551` 尚未开始。

2026-05-22 更新：oracle array `274549` 仍在运行；已完成 12/20 个 array task，0 个 Slurm 失败。完成样本为 `8dd7`、`8x9s`、`7vla`、`8v7l`、`7zdf`、`8v6v`、`8p71`、`8wpf`、`8pmd`、`7n70`、`7zdl`、`8wox`，样本级 summary 合计 `num_variants=240`、`num_jobs=1120`、`num_success=1120`。未完成但正在运行的样本为 `7nnl`、`7tju`、`8umt`、`8vm0`、`6bk8`、`7ut7`、`8x0b`、`7v19`。`6bk8` 的 partial variant summary 已显示 Rosetta job 级失败，典型错误为 `pdb_UNK` 缺少 RamaPrePro 主链 score table。realistic 仍未开始。

2026-05-23 更新：发现并修复了资源配置中的关键性能 bug。当前正在运行的旧 oracle array task 每条申请 8 核，但实际按单活跃 CPU 核串行执行 Rosetta jobs；新的 sbatch 默认已改为每 task 申请 1 核。本轮新增逐 job 时间成本统计，基于当时已完整的 18 个样本得到成功 Rosetta job 的 P75 为 `331.1 s/job`，P90 为 `390.6 s/job`。当前 oracle 成本公式应按真实 ligand 数写为 `20*n*(n+1)`，不是直接使用预测管线的 `nk`。`7v19` 的完整成本为 `1440` jobs，剩余成本按 P75/P90 约 `105.9/125.0` 小时；因此已取消只属于 Codex 的 `274549_19`，保留其 partial 输出，并将本轮完整分析集合定义为 `easy19_fast_validation`。`8x0b` 已完成，`7ut7` 尚余 `216` 个已审计未完成 jobs，预计剩余不超过约 `19.9` 小时，P90 上界约 `23.4` 小时。旧 realistic run `280768/280769` 也受相同性能 bug 影响，已在早期取消并由 1-CPU array run `281985` / 汇总 `281986`（run_id `20260523_realistic_easy20_instancef1_nstruct5_array1_fixed_v1`）取代。21:03 heartbeat 时该修复 run 已完整完成 5 个样本并累计 `60/60` Rosetta jobs 跑通，后续仍需完成 assignment 与 RMSD 评价。

2026-05-23 再更新：用户决定不把 `7v19` 永久排除，而是首先修复样本内部并行能力并用它作为真实性能验证。现已在代码中加入样本级 Rosetta 并行队列，覆盖 oracle 跨 variant 及 realistic 的全部 ligand-site-receptor jobs，并隔离并发失败日志。同步后将以 24 核全量覆盖 `7v19`；其串行 P75/P90 预算约 `132.4/156.3` CPU-core 小时，若 24 路并行有效，理想 wall time 约 `5.5/6.5` 小时，实际先按 `6-10` 小时预估。验证成功后，easy19 只是诊断过程中的临时状态，最终回到完整 easy20。

2026-05-23 21:47+08:00 更新：首次并行验证 `283182` 没有进入运行时间比较阶段，而是暴露了 partial 覆盖恢复 bug：参数生成不接受已存在的旧输出。这个失败没有推翻并行设计，只表明全量覆盖前必须清空获准覆盖的单一样本目录；同时它揭示批处理进程曾错误地把样本失败返回为 Slurm 成功，现已修复退出码传播。下一次提交将先只清理 `7v19` 目录，再以同一 24 核配置获得真实性能反馈。

2026-05-23 21:52+08:00 更新：已严格限定目标并清空原 run 的 `samples/7v19` 后提交重跑 job `283183`。它已获得 24 CPU 并开始重建样本输入，首次检查时输出目录已有 899 个文件，说明已越过上次的非幂等参数准备失败。仍需等待 Rosetta 进程运行和 CPU 时间增长后，才能宣布样本内部并行有效。

2026-05-23 21:53+08:00 更新：`283183` 已给出首个真实性能反馈：墙钟 `1:37` 内累计 CPU 时间 `17:28`，约 `10.8x` 并行消耗，说明公共执行池确实开始运行多个 Rosetta 子进程。当前还要再确认首批 scorefile/decoy 正常写出且 stderr 没有并行新故障；满足后将把 easy20 恢复为主集合，并处理 `7ut7` 的旧串行尾部。

2026-05-23 21:56+08:00 更新：`283183` 在墙钟 `6:26` 时累计 CPU 已达到 `2:12:50`（约 `20.6x`），且 stderr、样本错误均为空；节点内存远未成为瓶颈。该反馈已足以证明并行实现真实有效，因此已取消旧 `7ut7` task `274549_17` 与旧 collector `274550`，提交 `7ut7` 16 核覆盖 job `283281` 及依赖两项覆盖的新最终汇总 `283282`。oracle 的最终分析集合正式恢复为完整 easy20。

2026-05-23 22:02+08:00 更新：`7v19` 的首批真实输出已落盘，`283183` 有 24 个 scorefile / 76 个 outputs 且 CPU/墙钟约 `21.7x`；`7ut7` 的 `283281` 也已启动并达到约 `9.5x` CPU/墙钟。并行能力与输出路径均得到服务器证据支持。按已有单 job 预算和首批落盘速度，oracle 两项覆盖的收尾预计由 `7v19` 支配，先按从提交起约 `6-10` 小时、即当前仍约小于 `10` 小时估计；完成后 `283282` 自动汇总完整 easy20。

2026-05-24 12:02+08:00 更新：oracle 已从“性能修复验证”进入“完整数据可评价”阶段。最终 summary 显示 20/20 样本完成、5800 jobs 中 5739 跑通；按任务拆分为 `true_center_identity 110/112`、`offset_center_identity 990/1008`、`true_center_hungarian 463/468`、`offset_center_hungarian 4176/4212`。所有失败均来自 `6bk8`，代表性日志为 Rosetta 内部零长度几何向量错误。400 条 Hungarian 输出均标记为 `dp_virtual`，但未给出是否匹配正确、也没有 pose RMSD；因此下一步必须是 evaluation，而不是把 98.95% 流程跑通率改称 docking 成功率。与此同时，realistic 1-CPU run 只完成 12/20 样本，尾部若不迁移到样本内部并行，最慢样本仍需约 8 至 9 天。

2026-05-24 13:26+08:00 更新：本轮完成了三个立即产生反馈的动作。其一，新 oracle evaluator 已输出修复前的严格基线，确认 true center/identity 条件也未让 pose 达到理想 RMSD，且 Hungarian matching 仅约一半 occurrence 正确。其二，`6bk8` 的 `UNK` receptor bug 已被小型 smoke 证实可修，权威样本覆盖与最终评价链已经提交。其三，realistic 六个长尾已从串行 task 替换为 4/9/11/14/13/14 CPU 的同 run 覆盖任务，早期 `sstat` 已显示多核真实工作；最终结论需等 collector 后按统一 evaluation 口径汇总。

2026-05-24 13:27+08:00 更新：heartbeat 没有发现新失败类型。`6bk8` 权威覆盖与 realistic 六个覆盖任务都在使用多核并保持空 stderr；realistic 已保留 13 个完成样本 summary。此时最有信息增益的动作仍是等待 `284665` 给出覆盖后的严格 oracle 指标，并继续监视 realistic 并行任务是否出现内存或 Rosetta 错误。

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
/home/penghongen/分子对接尝试/pipeline_runs/20260523_realistic_easy20_instancef1_nstruct5_array1_fixed_v1
```

## Plan of Work

第一步，用固定同步脚本把本地 `Docking` 更新到服务器。

第二步，提交 `easy20_prescan_cpu.sbatch`。它只做统计，不运行 Rosetta，应该很快完成。完成后读取 `manifest.json`、`easy20_by_nk.csv`、`easy20_by_instance_f1.csv`，检查样本数量、`instance_f1` 是否读到、`nk` 是否合理、糖类/大 ligand/内部金属比例是否有异常。

第三步，若 prescan 表合理，提交 `oracle_easy20_cpu.sbatch` 和 `realistic_easy20_cpu.sbatch`。两者总 CPU 不能超过 96；如果同时跑，应各自降低 `JOBS` 或排队提交。若资源不足，优先跑 oracle，因为它直接回答下游上限问题。

2026-05-21 修订：CPU 任务不再使用 96 核单 job；本轮已运行任务采用每 task 8 核、最多 12 task 并发的 array，`--shard-id` 写入独立 shard summary，最终由 `collect_easy20_summary_cpu.sbatch` 汇总。

2026-05-23 修订：Slurm `TotalCPU` 与 wall time 证明本轮每个 8 核 task 实际仅使用约 1 个活跃核。对串行样本，后续提交改用 `#SBATCH --cpus-per-task=1` 与 array 并发控制；对长尾或需尽快完成的样本，现已实现 `--rosetta-jobs` 样本内部并行并先在 `7v19` 上真实监视验证。CPU 推荐数按 `ceil(计划 jobs / 40)` 取整，必要时可参考 `/30` 更积极分配，但总申请不得超过 96 核。

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
