# 批量分子对接与 instance 后处理调参

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

本计划遵守用户确认的 `.skill` 中 ExecPlan 工作模式。当前仓库没有根目录 `PLANS.md`，因此本文档自身包含完整上下文、实施步骤、验收标准与恢复方式。

## Purpose / Big Picture

本轮工作要把已经跑通的 Pocket Plus 下游分子对接 smoke 流程，扩展成一个可复现、可审计、可批量统计的 mini-benchmark v0。完成后，用户可以在服务器上针对 `/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/` 中当前可发现的样本目录，使用 CPU sbatch 与 joblib 并行运行 Rosetta `GALigandDock`，并得到每个样本、每个 site-ligand pair、每个 receptor 来源、每个 `nstruct` decoy 的结果表和汇总报告。

这轮同时要处理一个关键语义差异：前置神经网络训练时把独立金属离子也当作正类，但 docking 阶段不需要也不能对接独立金属离子。因此推理得到的 instance 不能全部信任。流程必须在 docking 前增加独立的 instance 后处理模块，支持最小体素数过滤、近邻 instance 合并、参数网格预扫描和审计记录。这个模块应尽量独立，方便未来复制回推理管线。

## Progress

- [x] (2026-05-19 00:00Z) 明确本轮采用 ExecPlan living document 模式，并把主计划文件放在 `Docking/分子对接记录/full80_docking_exec_plan.md`。
- [x] (2026-05-19 00:00Z) 用户确认本轮边界：本地只编辑 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking`；服务器只写 `/home/penghongen/分子对接尝试`；代码同步使用 `C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat`；一轮任务最多使用 48 核 CPU 与 2 张 A100。
- [x] (2026-05-19 00:00Z) 用户确认 full80 样本范围由服务器推理根目录当前可发现的样本目录决定，每次运行必须把实际样本清单落盘。
- [x] (2026-05-19 00:00Z) 实现 `instance_postprocess.py`，提供最小体素数过滤、保守合并、合并候选审计与 job 数估计。
- [x] (2026-05-19 00:00Z) 改造 runner，使其支持 run_id 隔离目录、经验参数、虚拟节点 matching、dry-run 与 scorefile 多 decoy 聚合。
- [x] (2026-05-19 00:00Z) 增加批处理 CLI、预扫描 CLI 与 CPU sbatch 模板，新增文件均放在 `Docking` 下。
- [x] (2026-05-19 00:00Z) 本地运行 `python -m compileall Docking` 通过；在 `Pocket_Plus_windows` 环境中完成 instance 后处理与虚拟节点 matching 的合成输入测试。
- [ ] 使用固定同步脚本同步到服务器。
- [x] (2026-05-19 00:00Z) 使用固定同步脚本同步到服务器，并完成服务器环境导入验证。
- [x] (2026-05-19 00:00Z) 在服务器允许目录完成 4 样本经验参数批次 `20260519_repro_known4_nstruct2`: 20/20 Rosetta job 跑通。
- [x] (2026-05-19 00:00Z) 在服务器允许目录完成 4 样本 raw-instance 批次 `20260519_repro_known4_raw_nstruct2`: 24/24 Rosetta job 跑通；这是去掉 7zdf round1 重复矩阵后的旧样本全矩阵复现。
- [x] (2026-05-19 00:00Z) 完成 full80 单参数预扫描：未截断预计 105250 Rosetta job，top5 预计 10440 job，top2 预计 4358 job，top1 预计 2186 job。
- [x] (2026-05-20 00:00Z) full80 top1 `nstruct=2` 原批次 `20260519_full80_top1_nstruct2` 已停止在 67 个样本完成、13 个样本 pending 的状态，并保留已有输出。
- [x] (2026-05-20 00:00Z) Slurm job `268093` 运行超过 18 小时后仍未收尾；已诊断为虚拟节点 assignment 在多 ligand 样本上出现大矩阵复杂度爆炸，取消该 job 以释放 48 CPU。
- [x] (2026-05-20 00:00Z) 已修复 `solve_assignment_with_virtual_nodes` 的大矩阵求解：小矩阵走 DP，大矩阵必须走 scipy Hungarian solver；如果服务器没有 scipy，应直接报错，不做近似 fallback。`AssignmentResult.solver` 会记录实际 solver。
- [ ] (2026-05-20 00:00Z) 剩余 13 样本批次 `20260520_full80_top1_pending13_scipy_required` 已提交为 Slurm job `268268`，当前仍在运行；中途审计显示 `8g4o`、`8utb`、`8wis` 3 个样本完成、10 个样本 pending、当前新 run Rosetta job `262/262` 跑通。
- [x] (2026-05-20 00:00Z) 完成一次中途审计并修复本地 Rosetta ligand residue 命名：`io_utils.read_ligand_candidates` 不再把唯一 `ATP/GDP` 等 CCD 名直接作为 `rosetta_name`，统一生成 `L01` 等三字符内部名，避免与 Rosetta 内置 residue type 冲突。
- [x] (2026-05-21 00:34+08:00) 先用 SSH key 检查 Slurm job `268268` 失败；随后根据用户补充的 `.skill` 密码完成只读服务器检查。
- [x] (2026-05-21 00:34+08:00) 确认 Slurm job `268268` 仍在运行，`sacct/squeue` 显示 `RUNNING`、48 CPU、节点 `cnode01`、已运行约 9 小时 55 分钟。
- [x] (2026-05-21 00:34+08:00) `20260520_full80_top1_pending13_scipy_required` 当前 13 个样本目录中 10 个有 summary、2 个 pending（`7w4g`, `8ca3`）、1 个 error（`8ypd`）；补跑 run 已完成样本的 Rosetta job 为 1062/1106 跑通，44 个失败 job 全部分布在 `6zku`。
- [x] (2026-05-21 00:34+08:00) 合并原 run 已完成部分和补跑已完成部分后，full80 top1 当前有 77/80 个样本有 summary，其中 57 个 `ok`、20 个 `skipped_no_sites_or_ligands`；已完成样本 Rosetta job 合计 1886/1950 跑通。
- [x] (2026-05-21 00:55+08:00) 本轮尝试再次只读检查 `268268` 与 pending13 run，但本地沙箱阻止 socket 连接；full80 top1 状态不能从本轮推进。
- [ ] 读取服务器审计结果，更新本计划的发现、结果和未完成项。

## Surprises & Discoveries

- Observation: 现有 `Docking/docking_pipeline/matching.py` 已实现 `solve_assignment_with_virtual_nodes`，但 `runner.py` 当前仍调用普通 `solve_assignment`。
  Evidence: `runner.py` 的 `_assign` 中使用 `assignments.append(solve_assignment(scores, site_ids, ligand_labels))`。

- Observation: 现有 `Docking/docking_pipeline/rosetta.py` 只解析 scorefile 的最后一条 `SCORE` 记录，不能保留 `nstruct>1` 的全部 decoy。
  Evidence: `io_utils.read_scorefile` 返回 `rows[-1]`，而用户要求 `nstruct=2` 默认并保留 best/mean/std 等聚合。

- Observation: `matching.py` 中 `_assignment_dp` 的实际返回顺序与调用方和 docstring 不一致，会导致虚拟节点 matching 解包失败，也会影响普通 assignment 的健壮性。
  Evidence: 合成测试调用 `solve_assignment_with_virtual_nodes` 时触发 `TypeError: 'float' object is not iterable`；已将 `_assignment_dp` 修正为返回 `(best_edges, best_cost)`。

- Observation: CPU sbatch 脚本不能在 conda activate 前启用 `set -u`，否则 conda 环境的 activate hook 会因为未绑定变量退出。
  Evidence: Slurm job `266337` 的 stderr 显示 `activate-binutils_linux-64.sh: line 68: ADDR2LINE: unbound variable`，作业在创建 pre-lock 前失败。已从 `Docking/sbatch/docking_cpu_batch.sbatch` 移除 `set -u`。

- Observation: Slurm `--export` 中的逗号分隔样本列表会被当成多个 export 项解析，导致 `SAMPLE_LIST=7zdf,8dd7,...` 实际只传入 `7zdf`。
  Evidence: Slurm job `266512` stdout 显示实际命令为 `--sample-list 7zdf`。后续改用允许目录中的样本列表文本文件。

- Observation: 保守 instance 后处理会改变旧 smoke 矩阵大小，不能直接称为旧结果复现。
  Evidence: Slurm job `266780` 使用默认 `min_voxels=30` 后，4 个已知样本 Rosetta job 全部跑通但总 job 数为 20；其中 `8pmd` 从旧文档的 12 job 变为 8 job，说明小 instance 被过滤。

- Observation: full80 若不限制每样本进入 docking 的 site 数，会产生不可接受的 Rosetta job 规模。
  Evidence: 单参数预扫描 `20260519_prescan_full80_conservative` 显示 80 样本 raw sites 2664、后处理 sites 1626、预计 Rosetta job 105250。top5 仍为 10440 job，top2 为 4358 job，top1 为 2186 job。

- Observation: top1 仍可能出现 assignment 后处理卡死，因为 `solve_assignment_with_virtual_nodes` 会把 `1 site x N ligands` padding 成 `N x N` 方阵并交给 bitmask DP。
  Evidence: Slurm job `268093` 运行 18:41:02 时仍有 13 个 pending 样本；多个 pending 样本已经写完大量 scorefile，例如 `6zgf` 有 168 个 scorefile、`8wlu` 有 150 个 scorefile，但没有样本级 summary。说明 Rosetta 已完成，卡点在后处理 assignment。

- Observation: 修复 solver 后，剩余 13 样本的长尾主要表现为 Rosetta job 数随 ligand 数放大，而不是 assignment 再次死锁。
  Evidence: Slurm job `268268` 运行约 2.7 小时时，新 run 已完成 `8g4o`、`8utb`、`8wis`，分别为 78/78、100/100、84/84 Rosetta job 跑通，完成样本 assignment solver 均记录为 `scipy_hungarian`；其余 pending 样本仍持续产生日志和 scorefile。

- Observation: 原 run 的样本级 `ok` 不能等价于 Rosetta job 全成功，必须在最终统计中保留 job 级失败列表。
  Evidence: 原 run 已完成部分共有 20 个失败 Rosetta job，分布在 `6bk8`、`7njo`、`8bly`、`8css`、`8uua`。归因包括 `pdb_UNK` residue 的 RamaPrePro 表缺失 2 个、Rosetta minimization 后 zero-length vector 15 个、`ATP` 内置 residue 与自生成 `ATP.params` 命名冲突 2 个、signal 11 崩溃 1 个。

- Observation: 复用唯一 CCD 名称作为 `rosetta_name` 会和 Rosetta 自带 residue type 发生缓存冲突。
  Evidence: `8css` 的 `ATP` true/cryoatom receptor 两个 job 失败日志显示 `Residue type ATP is already in the ResidueTypeSetCache`。本地已改为所有 ligand 均使用 `L01` 等内部 Rosetta 名称，审计标签仍保留原始 `ccd_id` 和 `label`。

- Observation: `268268` 尚未完成，但补跑 run 已从 3 个 completed 推进到 10 个 completed。
  Evidence: 密码登录后只读查询显示 `sacct/squeue` 仍为 `RUNNING`；`20260520_full80_top1_pending13_scipy_required` 的 completed 样本为 `6zgf`, `6zku`, `7mtc`, `7tb8`, `7z7s`, `8g4o`, `8utb`, `8wis`, `8wlu`, `8xh9`，pending 为 `7w4g`, `8ca3`，error 为 `8ypd`。

- Observation: 补跑 run 的 assignment solver 字段已经按修复预期记录为 `scipy_hungarian`。
  Evidence: `20260520_full80_top1_pending13_scipy_required` 已完成样本的 assignment 记录中 `solver_counts` 为 `scipy_hungarian: 30`；原 run 的历史 assignment solver 为空字段，共 141 条，说明旧输出不能反推 solver。

- Observation: 当前新增失败集中在 `6zku`，不是全局性 solver 死锁。
  Evidence: 补跑 run 已完成部分共有 1106 个 planned/result jobs，1062 个跑通、44 个失败；44 个失败 job 全部来自 `6zku`，returncode 为 `-11` 34 个、`1` 10 个，stderr tail 均归类为 Rosetta crash/internal error。

- Observation: 本轮没有拿到新的 full80 top1 远端反馈。
  Evidence: 密码 SSH 在本地 socket 创建阶段失败，错误为 `PermissionError: [WinError 10013] 以一种访问权限不允许的方式做了一个访问套接字的尝试。`；因此 `268268` 是否已完成、`7w4g`/`8ca3` 是否收尾、`8ypd` 是否仍为 error 都需要下一轮重新核对。

## Decision Log

- Decision: full80 第一轮默认使用 `nstruct=2`，可疑或代表样本后续使用 `nstruct=5` 或 `10` 重跑。
  Rationale: `nstruct` 是同一输入 job 独立生成的 Rosetta decoy 数，不是输入 PDB model 数。默认 2 能比 smoke test 多一点采样，又不会过早放大计算成本。
  Date/Author: 2026-05-19 / 用户与 Codex

- Decision: 真实 ligand 坐标只用于 `evaluate_with_truth` 风格的后验评估，不用于 docking 输入、site 选择、matching cost 或 best pair 决策。
  Rationale: 防止 ground truth 泄漏，保证 pipeline 统计口径可解释。
  Date/Author: 2026-05-19 / 用户与 Codex

- Decision: instance 后处理参数调优纳入正式范围，但采用分层调参。A 轨用保守经验参数先跑 full80 docking；B 轨做不跑 Rosetta 的预扫描网格，再决定是否补跑更优参数。
  Rationale: CPU 资源相对充足，但结果解释能力比算力更稀缺。预扫描能先缩小参数空间。
  Date/Author: 2026-05-19 / 用户与 Codex

- Decision: 独立金属离子在 docking 阶段应视为 non-dockable，不应作为 Rosetta ligand 候选；预测到金属的 instance 可能是推理正类正确，但对 docking 是假阳性或需过滤对象。
  Rationale: 前置网络训练正类定义和下游 docking 可对接对象定义不一致。
  Date/Author: 2026-05-19 / 用户与 Codex

- Decision: 批处理输出按 `run_id` 隔离，目录形态为 `/home/penghongen/分子对接尝试/pipeline_runs/{run_id}`。
  Rationale: 本轮会有经验参数、预扫描参数和可能的 `nstruct=5/10` 重跑，必须避免覆盖。
  Date/Author: 2026-05-19 / 用户与 Codex

- Decision: 旧矩阵复现必须使用 raw instance 口径，即 `min_voxels=0` 且禁用 merge；经验参数 full80 才使用保守后处理默认值。
  Rationale: 后处理本身会改变 site 数，不能把后处理后的 20/20 与旧文档的 28/28 混成同一个复现口径。
  Date/Author: 2026-05-19 / Codex

- Decision: 第一轮 full80 端到端 docking 使用 `max_sites_per_sample=1`、`min_voxels=30`、保守 merge 和 `nstruct=2`。
  Rationale: 不截断或 top5/top2 的 job 数过大，不适合作为第一轮全样本 smoke。top1 保留 80 样本覆盖和所有可对接 ligand，但把 Rosetta job 数控制到约 2186。
  Date/Author: 2026-05-19 / Codex

- Decision: 取消卡住的 Slurm job `268093`，保留已完成样本输出，并用修复后的 matching 代码另开 run_id 跑剩余 pending 样本。
  Rationale: 原 job 已占用 48 CPU 超过 18 小时且没有样本级推进；继续等待不会解决算法复杂度问题。新 run_id 可避免覆盖原始部分结果。
  Date/Author: 2026-05-20 / Codex

- Decision: 在 full80 全部完成前先做中途审计，并修复已经能明确归因的代码问题；最终 full80 汇总等剩余样本完成后再做一次。
  Rationale: 单样本耗时近似随选中 site 数、ligand 数和 receptor 数放大，运行时间呈长尾分布。等待最慢样本会拖慢对已暴露问题的处理。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: Rosetta `rosetta_name` 永远使用流程内部三字符名，不再在唯一 CCD 时复用真实 CCD 名。
  Rationale: `ccd_id` 和 `label` 已足够保留生物学身份；`rosetta_name` 的首要职责是避免 Rosetta residue type 冲突并保证 params/complex 输入可运行。
  Date/Author: 2026-05-20 / Codex

## Outcomes & Retrospective

2026-05-20 中途审计结论：同意用户关于长尾复杂度的判断。当前不应为了等待全部样本完成而暂停分析；已经完成和半完成样本足以确认两个问题：一是大矩阵 assignment 需要强制走 `scipy_hungarian`，二是 Rosetta 内置 residue 名称冲突需要在 ligand 命名层面修复。当前本地代码已修复第二点，但尚未同步服务器；等待 `268268` 完成或停止后再同步，避免影响正在运行的批次。

2026-05-21 更新：根据用户补充的服务器密码，已完成只读服务器核对。`268268` 仍在运行，不能启动会超出资源上限的重型新任务。当前 full80 top1 已有 77/80 个样本有 summary，剩余 `7w4g`、`8ca3` pending，`8ypd` error。已完成样本的 Rosetta job 跑通率为 1886/1950，其中原 run 824/844，补跑 run 1062/1106。补跑 run 的 solver 字段已确认是 `scipy_hungarian`。

2026-05-21 00:55+08:00 更新：本轮服务器状态未能核对，原因是当前本地沙箱阻止 SSH socket。不要据此改写 77/80、1886/1950 或 pending/error 统计；这些仍沿用 00:34+08:00 的最后可信远端观察。下一次服务器可访问时，第一优先级仍是确认 `268268` 是否完成并合并两段 full80 top1。

## Context and Orientation

项目根目录是 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus`。本轮只允许编辑本地 `Docking` 目录。服务器运行产物只能写入 `/home/penghongen/分子对接尝试`。服务器可只读访问 Pocket Plus 项目、Rosetta、推理结果、ligand 数据和结构数据。

当前 docking 入口说明在 `Docking/readme.md`。已有阶段报告在 `Docking/分子对接记录/progress1_readable.md` 和 `Docking/分子对接记录/progress1.md`。已跑通的流程成功不是 docking 几何精度成功，而是 Rosetta 进程结束、输出 PDB 与 scorefile 存在、分数字段可解析。

现有 Python 骨架在 `Docking/docking_pipeline/`：

- `config.py` 定义服务器路径、Rosetta 参数和 matching 参数。
- `records.py` 定义 `InferenceSite`、`LigandCandidate`、`DockingJob`、`DockingResult`、`PairScore`、`AssignmentResult`。
- `io_utils.py` 读取推理 JSON/NPZ、ligand mapping、分辨率表、scorefile，并转换 receptor CIF 到 PDB。
- `rosetta.py` 生成 params、complex PDB、Rosetta XML 和命令，并运行 Rosetta job。
- `shape_scoring.py` 计算当前径向 shape proxy。
- `matching.py` 已含普通矩形 assignment 与虚拟节点 assignment。
- `runner.py` 编排单样本 smoke pipeline。

术语定义：

- `instance` 是前置神经网络在体素空间预测出的一个 ligand-like 连通区域。
- `dockable_ligand` 是 docking 阶段应对接的小分子候选。
- `non_dockable_metal` 是独立金属离子；它在训练时可能是正类，但不进入 Rosetta ligand docking。
- `nstruct` 是 Rosetta 对同一个输入 job 生成的独立 decoy 数。
- `run_id` 是一次批处理或预扫描运行的唯一名字，用于隔离输出目录。

## Plan of Work

第一步，新增 `Docking/docking_pipeline/instance_postprocess.py`。该模块定义 `InstancePostprocessOptions` 和 `PostprocessedSites` 等轻量结构，提供按最小体素数过滤、两两 instance 合并候选分析、保守合并执行、参数网格预扫描所需的函数。函数输入应以 `InferenceSite`、`instance_label`、`origin`、`voxel_size` 为主，不依赖 Rosetta。输出要包含后处理后的 site 列表、被过滤项、合并项、候选合并报告和参数值，便于写入审计 JSON。第一版合并使用最近体素距离、中心距离、体素数比例与合并后包围盒跨度变化作为可解释规则。

第二步，扩展 `io_utils.py` 和 `rosetta.py`。`io_utils` 需要提供读取 scorefile 所有 decoy 行的函数，保留原 `read_scorefile` 兼容旧路径。`rosetta.py` 的 `run_rosetta_job` 应保存 `score_rows` 和 decoy 聚合信息，默认 cost 使用 best decoy，同时保留 mean/std。`records.py` 中的 `DockingResult` 应增加适合 JSON 化的字段或方法。

第三步，改造 `runner.py`。新增可以指定 `run_root`、`run_id`、后处理参数、是否启用虚拟节点、是否只 dry-run 的入口。保留旧 `run_sample_smoke` 兼容单样本调用，但内部走新入口。新入口要把每个样本写入 `pipeline_runs/{run_id}/samples/{pdb_id}`，把配置、原始 instance、后处理 instance、ligand 候选、Rosetta jobs、results、assignment 写入 `audit/`。

第四步，新增批处理与预扫描 CLI。建议文件：

- `Docking/run_docking_batch.py`：发现样本、保存样本清单、用 joblib 并行运行样本级 pipeline。
- `Docking/run_instance_prescan.py`：不跑 Rosetta，只遍历参数网格，输出 instance 统计和估计 job 数。
- `Docking/sbatch/docking_cpu_batch.sbatch`：CPU sbatch 模板，使用服务器 conda 环境 `Pocket_Plus_centos7_cu121_allgpu`，限制 CPU 线程，调用批处理 CLI。

第五步，本地验证。运行 `python -m compileall Docking` 或等价命令，必要时加轻量测试脚本验证 assignment、scorefile parsing、instance postprocess 不依赖服务器数据的部分。

第六步，同步服务器并运行。只能使用 `C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat`。服务器上只在 `/home/penghongen/分子对接尝试` 下写入。先复现 7zdf、8dd7、8x9s、8pmd，再提交 full80 经验参数 `nstruct=2` 批处理与不跑 Rosetta 的 instance 预扫描。CPU 总核数不超过 48。

第七步，读取结果并更新报告。汇总样本数、可运行样本数、无可对接 ligand 样本、Rosetta job 跑通率、decoy 聚合、普通/虚拟 matching 结果、instance 过滤和合并统计、预扫描参数候选。把重要结论更新到本 ExecPlan 与后续报告。真正的 docking 成功率不在本计划内计算，必须交给 evaluation 计划基于 GT、RMSD 和 top k/top k% 单独统计。

## Concrete Steps

在本地 PowerShell 中编辑和验证：

    cd C:\Users\15919\OneDrive\My_Project\Pocket_Plus
    python -m compileall Docking

同步服务器：

    C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat

服务器侧预期命令形态，所有输出都写入 `/home/penghongen/分子对接尝试/pipeline_runs/{run_id}`：

    cd /home/penghongen/My_Project/Pocket_Plus
    source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
    python Docking/run_docking_batch.py --run-id 20260519_repro_known_nstruct2 --sample-list 7zdf,8dd7,8x9s,8pmd --nstruct 2 --jobs 8
    python Docking/run_instance_prescan.py --run-id 20260519_instance_prescan_full80 --jobs 24

full80 CPU sbatch 提交应使用 `Docking/sbatch/docking_cpu_batch.sbatch`，并通过环境变量或 CLI 参数指定 run_id、jobs、nstruct 和样本来源。实际提交命令和 job id 会在运行后补充到本节。

## Validation and Acceptance

本地验收：

- `python -m compileall Docking` 成功，无语法错误。
- 新增模块可以导入，不依赖服务器数据即可完成小型合成输入测试。
- 旧入口 `Docking/run_docking_sample.py --pdb-id 7zdf` 的参数和调用方式仍可用。

服务器验收：

- 复现样本运行后，`pipeline_runs/{run_id}/samples/7zdf/audit/summary.json` 存在，并记录 `num_jobs`、`num_success`、`assignments`、`postprocess`、`rosetta_options`。
- full80 批处理将实际发现的样本写入 `pipeline_runs/{run_id}/config/sample_list.txt`。
- 每个样本的失败不会终止整个批处理，失败原因进入样本级 audit。
- `nstruct=2` 时，每个 Rosetta job 保留所有 decoy rows，并在 pair cost 中使用 best decoy，同时报告 mean/std。
- 虚拟节点 matching 的 `unmatched_sites`、`unmatched_ligands` 和 `virtual_edges` 出现在审计结果中。
- instance 预扫描输出不同参数组合下的 raw/postprocessed instance 数、过滤数、合并数和预计 Rosetta job 数。

## Idempotence and Recovery

本地文件修改是普通文本与 Python 代码修改，可以通过 git diff 检查。不要回滚用户已有修改。

服务器输出按 run_id 隔离，重复运行同一个 run_id 默认应能 resume 或覆盖同一 run_id 下的审计文件，但不能写出 `/home/penghongen/分子对接尝试`。若需要重新跑不同参数，创建新的 run_id。Rosetta job 失败时保留 stdout/stderr 和中间输入，批处理继续处理其他样本。

如果 sbatch 作业失败，使用对应 run_id 下的 `logs/` 与样本级 `audit/error.json` 定位问题。不要删除只读数据源，不要修改 Rosetta 安装、推理目录或 ligand 数据目录。

## Artifacts and Notes

当前已知服务器输入：

    /home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy
    /storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv
    /storage/penghongen/EMDB_PDB_resolution_3.5.csv
    /storage/chenzhaoyang/cryo_em/CIF_3.5_atom
    /storage/chenzhaoyang/cryo_em/result_split

当前已知 Rosetta：

    /home/penghongen/software/rosetta/bin/rosetta_scripts_430
    /home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main/database
    /home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main/source/scripts/python/public/molfile_to_params.py

## Interfaces and Dependencies

新增或稳定接口：

- `InstancePostprocessOptions`: instance 后处理参数，至少包含 `min_voxels`、`merge_min_voxel_distance`、`merge_center_distance`、`merge_max_bbox_span_increase`、`enable_merge`。
- `postprocess_sites(sites, label, origin, voxel_size, options) -> PostprocessResult`: 返回后处理 site、过滤记录、合并记录和候选记录。
- `run_sample(...) -> dict[str, object]`: 新样本级入口，支持 run_root/run_id、`nstruct`、后处理、虚拟节点和审计落盘。
- `run_docking_batch.py`: CLI 入口，支持 `--run-id`、`--sample-list`、`--sample-root`、`--nstruct`、`--jobs`、`--dry-run`。
- `run_instance_prescan.py`: CLI 入口，支持参数网格和 full80 样本发现。

依赖优先使用 Python 标准库、NumPy、joblib 和现有项目工具。不要引入新的重型依赖；第一版调参使用网格扫描，不引入 Optuna。

## Revision Notes

2026-05-19: 初始 ExecPlan 落盘。根据用户 grill 过程中确认的决策，加入 full80、instance 后处理、双轨调参、`nstruct=2`、真实坐标只后验评估、run_id 隔离和服务器写入边界。

2026-05-19: 完成第一批本地实现。新增 instance 后处理模块、批处理入口、预扫描入口、CPU sbatch 模板，改造 runner/rosetta/io_utils/records/matching 以支持 run_id、虚拟节点和多 decoy 审计。

2026-05-19: 首次复现 sbatch job `266337` 在 conda activate 阶段失败，原因是 `set -u` 与 conda hook 中未绑定变量冲突。已修订 sbatch 模板，准备重新同步并提交。

2026-05-19: 服务器侧确认推理根目录当前可发现 80 个样本。预扫描 smoke `20260519_prescan_smoke3` 对 3 个样本、2 个参数组合跑通并写出 6 行结果。4 样本经验参数批次 `266780` 跑通 20/20，但因 `min_voxels=30` 改变了 8pmd 矩阵，需另跑 raw-instance 复现。

2026-05-19: 完成 raw-instance 4 样本复现 `20260519_repro_known4_raw_nstruct2`，去掉旧文档中 7zdf round1 重复矩阵后为 24/24 Rosetta job 跑通。完成 full80 预扫描并选择 top1 作为第一轮全样本 docking 口径。Slurm job `268093` 已启动，5 分钟时已有 35 个样本目录、11 个样本完成或跳过、0 个失败。

2026-05-20: 诊断并修复 full80 top1 卡点。`268093` 在 18 小时后仍未完成，已完成 67 个样本 summary、0 个样本级失败、partial Rosetta 跑通率 824/844；剩余 13 个样本中多个已经完成 Rosetta 输出但卡在虚拟节点 assignment。已取消该 job。根据用户要求，移除无 scipy 时的贪心兜底，改为直接报错，并在 assignment 审计中写入实际 solver。

2026-05-20: 根据用户反馈修正术语边界。本计划只统计流程跑通、Rosetta job 跑通、solver 和失败日志，不再把这些称为对接成功；真正成功率由新的 evaluation 计划基于 GT center hit、Hungarian match accuracy、RMSD 和 top k/top k% 计算。
