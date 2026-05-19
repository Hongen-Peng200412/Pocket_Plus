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
- [x] (2026-05-19 00:00Z) 在服务器允许目录完成 4 样本经验参数批次 `20260519_repro_known4_nstruct2`: 20/20 Rosetta job 成功。
- [x] (2026-05-19 00:00Z) 在服务器允许目录完成 4 样本 raw-instance 批次 `20260519_repro_known4_raw_nstruct2`: 24/24 Rosetta job 成功；这是去掉 7zdf round1 重复矩阵后的旧样本全矩阵复现。
- [x] (2026-05-19 00:00Z) 完成 full80 单参数预扫描：未截断预计 105250 Rosetta job，top5 预计 10440 job，top2 预计 4358 job，top1 预计 2186 job。
- [ ] full80 top1 `nstruct=2` docking 批次 `20260519_full80_top1_nstruct2` 已提交为 Slurm job `268093`，正在运行。
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
  Evidence: Slurm job `266780` 使用默认 `min_voxels=30` 后，4 个已知样本全部成功但总 job 数为 20；其中 `8pmd` 从旧文档的 12 job 变为 8 job，说明小 instance 被过滤。

- Observation: full80 若不限制每样本进入 docking 的 site 数，会产生不可接受的 Rosetta job 规模。
  Evidence: 单参数预扫描 `20260519_prescan_full80_conservative` 显示 80 样本 raw sites 2664、后处理 sites 1626、预计 Rosetta job 105250。top5 仍为 10440 job，top2 为 4358 job，top1 为 2186 job。

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

## Outcomes & Retrospective

本节将在主要里程碑完成后更新。当前尚未完成实现与服务器运行。

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

第七步，读取结果并更新报告。汇总样本数、可运行样本数、无可对接 ligand 样本、Rosetta job 成功率、decoy 聚合、普通/虚拟 matching 结果、instance 过滤和合并统计、预扫描参数候选。把重要结论更新到本 ExecPlan 与后续报告。

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

2026-05-19: 服务器侧确认推理根目录当前可发现 80 个样本。预扫描 smoke `20260519_prescan_smoke3` 对 3 个样本、2 个参数组合成功写出 6 行结果。4 样本经验参数批次 `266780` 成功完成 20/20，但因 `min_voxels=30` 改变了 8pmd 矩阵，需另跑 raw-instance 复现。

2026-05-19: 完成 raw-instance 4 样本复现 `20260519_repro_known4_raw_nstruct2`，去掉旧文档中 7zdf round1 重复矩阵后为 24/24 成功。完成 full80 预扫描并选择 top1 作为第一轮全样本 docking 口径。Slurm job `268093` 已启动，5 分钟时已有 35 个样本目录、11 个样本完成或跳过、0 个失败。
