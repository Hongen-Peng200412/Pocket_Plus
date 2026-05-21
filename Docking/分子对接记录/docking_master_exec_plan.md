# 分子对接评估与打分总控 ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

本计划是 `Docking` 目录下本轮长时间连续工作的总控计划。它统领三个互相协同但边界清晰的方向：Rosetta 流程跑通、真实几何评估、以及用于 Hungarian 匹配前打分的机器学习或网格调参。本仓库当前没有根目录 `PLANS.md`，因此本文档自身包含完整上下文、边界、实施步骤、验收标准和恢复方式。

## Purpose / Big Picture

本轮工作的目标不是证明 Rosetta job 能结束，而是回答“Pocket Plus 预测位点经过 docking 和 Hungarian 匹配后，是否真的找到了正确 ligand，并且 pose 是否接近真实结构”。完成后，用户可以看到每个样本、每个预测位点、每个 ligand、每个 receptor 来源下的分层评估表：前置 instance center 是否命中真实 ligand center，最终 docked pose 的 RMSD 是否达到 2/3/5 Å 阈值，真实匹配在当前打分矩阵中排第几、是否进入 top k 或 top k%，以及哪些 ligand 类型、样本条件、预测质量或 receptor 来源导致失败。

本计划必须保持一个重要术语边界：`Docking` 现有批处理中的“成功”只能叫“流程跑通”或“Rosetta job 跑通”，意思是进程结束、scorefile/PDB 可解析、审计文件存在；真正的“对接成功”必须由 evaluation 计算，包括前置位点命中、Hungarian 匹配正确率、RMSD 和 top k/top k% 接近程度。

## Progress

- [x] (2026-05-20 13:30Z) 用户确认需要一个总控 ExecPlan，统领 docking 跑通、evaluation、feature table、ML scoring 和服务器调度。
- [x] (2026-05-20 13:30Z) 用户确认本地可编辑范围是 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking`，服务器可写范围是 `/home/penghongen/分子对接尝试`。
- [x] (2026-05-20 13:30Z) 用户确认同步只能使用 `C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat`，服务器同时最多使用 96 核 CPU 和 2 张 A100。
- [x] (2026-05-20 13:30Z) 用户确认 Codex 可以取消自己提交的 Slurm job，但绝不能取消或修改用户自己提交的任务。
- [x] (2026-05-20 13:30Z) 用户确认不必等待 full80 全部完成，可以在已有大多数 Rosetta 结果上并行启动 evaluation、bug 排查、feature table 和 ML scoring。
- [x] (2026-05-20 13:45Z) 修正 `full80_docking_exec_plan.md` 中“成功”相关表述，把 docking runner 层统一改为“流程跑通”语义。
- [x] (2026-05-20 13:45Z) 新建 evaluation 子 ExecPlan，覆盖 GT 构建、前置 center hit、RMSD、top k/top k% 和分层统计。
- [x] (2026-05-20 13:45Z) 新建 ML scoring 子 ExecPlan，覆盖 deployable/oracle 特征边界、极简网格调参、tree ensemble/ranking model 和 assignment-level objective。
- [x] (2026-05-20 13:50Z) 实现 evaluation 主干脚本第一版，能消费已有 completed/partial run audit，输出 truth、pred site、pose、assignment、center hit、RMSD 和 rank 表。
- [x] (2026-05-20 14:05Z) 使用固定同步脚本同步本地 `Docking` 到服务器，并在服务器跑通 `20260520_eval_smoke3` 小样本 evaluation smoke。
- [x] (2026-05-20 14:15Z) 在服务器允许目录完成第一版 partial evaluation `20260520_eval_full80_partial_v1`，输出 75 个已完成/跳过样本的基础评估表。
- [x] (2026-05-20 14:20Z) 新增 `Docking/sbatch/evaluation_cpu.sbatch`，规定后续可能超过几分钟的 evaluation、feature table 或 ML 任务必须通过 CPU sbatch 与 heartbeat 检查推进。
- [x] (2026-05-21 00:34+08:00) 本地完成指标口径复核，并新增 `Docking/分子对接记录/docking_metrics_readable.md` 作为用户可读指标解释文档。
- [x] (2026-05-21 00:34+08:00) 根据用户补充的 `.skill` 密码完成只读服务器检查：`268268` 仍在运行，补跑 run 已完成 10/13 个样本，2 个 pending，1 个 error。
- [x] (2026-05-21 00:34+08:00) 读取远端 `best_summary.json`，确认 `avg_instance_f1=0.36896760343806495`，该值仍是体素连通域覆盖口径，不是前置 center hit。
- [x] (2026-05-21 00:55+08:00) 本轮尝试用密码 SSH 只读检查 `268268` 与远端 run 目录，但当前本地沙箱阻止 socket 连接；不要把本轮视为确认 job 完成或失败。
- [x] (2026-05-21 00:55+08:00) 在本地 evaluation 代码中补齐前置 center Hungarian precision/recall，保留原 loose hit 口径，并通过 `python -m compileall Docking` 与合成样例验证。
- [x] (2026-05-21) 用户指定下一大轮方向：统计两套 easy20，并在 `easy20_by_nk` 上做四类 oracle docking，在 `easy20_by_instance_f1` 上做 realistic docking。
- [x] (2026-05-21) 新建 `oracle_easy20_exec_plan.md` 与 `oracle_easy20_experiment.md`，实现 easy20 prescan、oracle runner 和三张 CPU sbatch 脚本。
- [ ] 后续需要同步 `evaluation_cpu.sbatch` 到服务器；若 `268268` 仍用 48 CPU，evaluation/feature/ML sbatch 任务最多再用 48 CPU，总 CPU 不超过 96。

## Surprises & Discoveries

- Observation: 现有文档和阶段汇报中有些“成功”实际指流程跑通，不是几何意义的对接成功。
  Evidence: `Docking/分子对接记录/progress1_readable.md` 已在前半部分说明“当前统计的是流程成功率，不是 docking 几何精度”，但 `full80_docking_exec_plan.md` 里仍有多处 `Rosetta job 成功`、`样本级 ok` 等容易被误读的表达。

- Observation: 原 full80 run 的样本级 `ok` 会掩盖 job 级失败，不能作为最终成功率。
  Evidence: 原 run `20260519_full80_top1_nstruct2` 已完成 67 个样本，但 Rosetta job 级为 824/844；失败日志包含 `pdb_UNK`、zero-length vector、`ATP.params` 命名冲突和 signal 11。

- Observation: full80 的单样本运行时间呈长尾分布，等待最后一个样本完成会推迟已经可以做的 evaluation 和 bug 修复。
  Evidence: `268268` 的 pending13 补跑中，较大的样本如 `8g4o`、`8utb`、`8wis` 各自有 78、100、84 个 Rosetta job；这些样本完成后已经足以验证 solver 记录和一批评估逻辑。

- Observation: 第一版 partial evaluation 已经证明“流程跑通”与“真实对接成功”差距很大。
  Evidence: `20260520_eval_full80_partial_v1` 读取两个 docking run，纳入 75 个有 summary 的样本、828 个 GT ligand instance、66 个 selected pred sites、1642 个 pose rows。前置 center hit rate 为 3 Å 5.07%、4 Å 6.04%、8 Å 8.33%；direct atom-order RMSD 有效 1618 行，RMSD <= 2 Å 为 0，<= 3 Å 为 0.31%，<= 5 Å 为 1.17%。

- Observation: 当前 rank/top k 第一版过粗，不能作为严格 assignment 成功率。
  Evidence: `20260520_eval_full80_partial_v1` 的 rank_summary 显示 truth_top1_rate 为 1.0，这是因为当前只把候选 ligand label 是否属于 GT label 集合作为 truth，而没有同时约束 pred site 与 GT occurrence，因此会高估排序接近程度。

- Observation: 推理搜索的 `best_summary.json` 与 docking evaluation 的前置 center hit 是不同口径。
  Evidence: `src/inference/main/two_stage_basic.py` 的 stage2 输出来自 `voxel_tuning.py`，其中 `avg_instance_f1` 汇总的是 `voxel_evaluator.evaluate_instance_mask` 的连通域覆盖 precision/recall/F1；docking evaluation 的 center hit 则是 selected pred site center 到 GT ligand center 的 3/4/8 Å 距离阈值。

- Observation: 服务器检查改用密码认证后拿到了新反馈，`268268` 不是完成态。
  Evidence: `sacct` 显示 job `268268`/`DockFull80` 为 `RUNNING`，48 CPU，节点 `cnode01`，已运行约 9 小时 55 分钟；`squeue` 同样显示 running。

- Observation: full80 top1 当前是“多数样本有 summary，但补跑尚未收尾”的中间态。
  Evidence: 原 run `20260519_full80_top1_nstruct2` 为 67 completed、13 pending、Rosetta job 824/844；补跑 run `20260520_full80_top1_pending13_scipy_required` 为 10 completed、2 pending（`7w4g`, `8ca3`）、1 error（`8ypd`）、Rosetta job 1062/1106。合并后当前 77/80 样本有 summary，已完成样本 job 1886/1950。

- Observation: 远端 inference `best_summary.json` 的数值进一步支持“不是 center hit”的判断。
  Evidence: `/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy/best_summary.json` 存在，`avg_num_candidates=28.3375`、`avg_instance_precision=0.4690510621159653`、`avg_instance_recall=0.47170958421751286`、`avg_instance_f1=0.36896760343806495`，`postprocess_params.threshold=0.97`、`min_component_voxels=10`。

- Observation: 本轮自动化环境无法建立服务器 socket，因此没有新的 Slurm 真值。
  Evidence: 密码 SSH 连接在本机创建 socket 时失败，错误为 `PermissionError: [WinError 10013] 以一种访问权限不允许的方式做了一个访问套接字的尝试。`；这不是 `268268` 的远端状态反馈。

- Observation: evaluation 现在可以同时输出 loose center hit 与严格一一匹配 center hit。
  Evidence: `Docking/evaluation/run_evaluation.py` 的 `evaluate_site_hits` 新增 `hungarian_recall_le_3A/4A/8A`、`hungarian_precision_le_3A/4A/8A` 和 solver 统计；合成样例中 2 个 GT、3 个 pred 得到 recall 1.0、precision 0.6666666666666666。

- Observation: 当前 Rosetta 初始中心语义适合做 true-center / offset-center oracle 实验。
  Evidence: `Docking/docking_pipeline/rosetta.py` 的 `translate_ligand_to_site()` 把 ligand PDB 的原子坐标均值平移到 `site.center_world_xyz`。

## Decision Log

- Decision: 本轮拆分为一个总控 ExecPlan 与至少三个子计划：docking runner、evaluation、ML scoring。
  Rationale: docking 跑通、真实评估、统计解释和 ML 打分的输入输出不同，混在一个文件中会让“流程跑通”与“对接成功”继续混淆。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: Evaluation 是主干，Feature/ML 是支线，Docking Runner 只负责流程跑通、失败归因、补跑和资源调度。
  Rationale: 没有 evaluation 就没有标签、RMSD 或 match accuracy；ML scoring 必须消费 evaluation 的标准 truth/evaluation tables，不能自己重新定义 GT。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 前置位点评估和最终 pose 评估分开统计。
  Rationale: 前置网络预测对了 center 但 Rosetta pose 失败、或者前置 center 偏了但 Rosetta 偶然给出低 RMSD，是两类不同问题，需要分别诊断。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 最终严格标准以 RMSD <= 2.0 Å 为主指标，同时报告 2.0/3.0/5.0 Å 三档，并额外报告真实匹配的 rank、top k、top k% 和正确 assignment cost 与当前最优 assignment cost 的比值或差值。
  Rationale: RMSD 阈值回答“是否足够接近”，rank/top k/top k% 回答“正确解在当前打分函数下离被选中有多近”，后者能指导打分函数和 ML 模型优化。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: ML scoring 同时走两条路线：少变量网格调参和先进 tree ensemble/ranking model。
  Rationale: 网格调参给出透明 baseline，tree ensemble 可直接利用非线性特征和交互项；用户明确建议可以直接走极端，不必过度保守。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: `gt_instance` 只能进入 label、objective、分层统计和 oracle analysis，不能进入最终可部署 scoring 模型特征；`pred_instance` 统计可以进入 deployable 特征。
  Rationale: 真实推理/对接场景中不可见 GT，把 GT 当特征会得到 oracle evaluator 而不是可部署 scorer。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 后续任何预计超过几分钟的服务器任务必须走 sbatch 与 heartbeat/automation 检查，不再用长时间 SSH 前台命令等待。
  Rationale: 前台长命令在 Codex 侧看起来像“无进展运行”，不利于长时间连续工作，也容易被中断。sbatch 能保留服务器进度、日志和锁文件。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 用户可读汇报按“推理后处理指标、Docking 流程指标、真实 evaluation 指标”三层拆开。
  Rationale: `avg_instance_f1`、Rosetta job 跑通率和 RMSD/top-k 分别回答不同问题；拆开定义能防止在错误指标上优化。
  Date/Author: 2026-05-21 / Codex

- Decision: 在服务器暂时不可读时，优先推进本地 evaluation 口径修正，而不是提交或等待任何新服务器任务。
  Rationale: 当前缺少 `268268` 的新状态，且资源边界要求不能盲目提交重型任务；center Hungarian precision/recall 是已知待办，能让下一次远端可访问时直接重跑更严格评估。
  Date/Author: 2026-05-21 / Codex

- Decision: 下一大轮新增 easy20 oracle / realistic 子计划，不把 oracle 逻辑塞进真实流程 runner。
  Rationale: oracle 实验需要 GT center 和 GT ligand identity，是诊断/上限分析；realistic 实验应继续走真实预测 site pipeline，避免可部署流程与 oracle 流程混淆。
  Date/Author: 2026-05-21 / 用户与 Codex

## Outcomes & Retrospective

当前处于总控计划落盘阶段。下一步要把已有 docking 计划的术语修正为“流程跑通”，并新建 evaluation 与 ML scoring 子计划。真正的阶段性成果应是：在 `268268` 全部完成之前，已经能对已有样本输出前置 center hit、RMSD、rank/top k/top k% 和分层统计表。

2026-05-20 更新：总控计划、evaluation 子计划、ML scoring 子计划已经落盘。`Docking/evaluation/run_evaluation.py` 第一版已经实现并通过本地 `python -m compileall Docking` 与 `Pocket_Plus_windows` 环境下的 `--help` 验证。下一步是同步服务器并在允许目录启动 partial evaluation run。

2026-05-20 再更新：已同步服务器并完成 `20260520_eval_full80_partial_v1`。这次结果只应视为第一版诊断信号，不是最终成功率：RMSD 用 direct atom-order 方法，rank/top k 尚未约束 GT occurrence。关键收获是前置 center hit 和最终 RMSD 都明显偏低，说明 evaluation 方向必须优先推进。

2026-05-21 更新：完成一次本地口径复核和文档推进。`best_summary.json` 的 `avg_instance_*` 不应再被称为 center hit；已新增 `docking_metrics_readable.md` 定义所有关键指标的分母、分子、阈值和 caveat。随后用用户补充的密码完成服务器只读检查：`268268` 仍在运行，补跑 run 已完成 10/13 个样本，合并后 full80 top1 当前 77/80 样本有 summary、已完成样本 Rosetta job 1886/1950 跑通。下一步应等 `268268` 收尾或下一次 heartbeat 继续检查，而不是现在提交新的重型任务。

2026-05-21 00:55+08:00 更新：本轮服务器 SSH 被本地沙箱 socket 权限拦截，因此 `268268` 状态没有新确认。为避免空转，已在本地实现前置 center Hungarian precision/recall；下一次服务器可访问时，应同步 Docking 后重跑 evaluation，把 loose hit 与 Hungarian precision/recall 同时写入 summary。

## Context and Orientation

项目根目录是 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus`。本轮本地只允许编辑 `Docking` 目录。服务器上只允许写 `/home/penghongen/分子对接尝试`，可以读取 Pocket Plus 项目、Rosetta、推理结果、ligand 数据和结构数据。代码同步只能使用 `C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat`。

现有 docking 代码在 `Docking/docking_pipeline/`。`runner.py` 负责样本级流程，`rosetta.py` 负责 Rosetta 命令和 scorefile 解析，`matching.py` 负责普通/虚拟 Hungarian assignment，`io_utils.py` 负责推理输出、ligand mapping、scorefile 和 receptor PDB 转换，`instance_postprocess.py` 负责 instance 最小体素过滤和合并。批处理入口是 `Docking/run_docking_batch.py`，CPU Slurm 模板是 `Docking/sbatch/docking_cpu_batch.sbatch`。

已有重要计划和记录：

- `Docking/分子对接记录/full80_docking_exec_plan.md`：当前 full80 docking 跑通、补跑、solver、失败日志的计划。
- `Docking/分子对接记录/progress1_readable.md`：早期 smoke test 和下一步建议，后半部分已经提出虚拟节点、shape score 局限和 ML 调参方向。
- `Docking/分子对接记录/7zdf_emerald_id_exec_plan.md`：7zdf 起步审计和中文路径经验。

当前服务器 run 语义：

- `20260519_full80_top1_nstruct2`：原 full80 top1 `nstruct=2` run。停止时 67 个样本完成、13 个 pending，Rosetta job 层面 824/844 跑通。
- `20260520_full80_top1_pending13_scipy_required`：剩余 13 样本补跑 run，使用 scipy Hungarian solver，正在逐步完成长尾样本。

术语定义：

- `流程跑通`：Rosetta 进程结束，scorefile/PDB 和审计文件可解析。它不表示 docking pose 正确。
- `前置位点命中`：预测 instance center 与真实 ligand instance center 的距离低于某个阈值，例如 3/4/8 Å。
- `最终 pose 成功`：docked pose 与真实 ligand pose 的 heavy-atom RMSD 低于阈值，主阈值为 2.0 Å，同时报告 3.0/5.0 Å。
- `rank/top k/top k%`：真实匹配在当前 cost 或 model score 排序下的位置。它衡量正确解离当前最优解有多近。
- `deployable 特征`：真实推理时可见的特征，例如 pred instance 分数、体素数、Rosetta score、pose 与预测 mask overlap、ligand 属性、receptor 来源、map 分辨率。
- `oracle 特征`：只在评估时可见的 GT 相关量，例如 GT center、GT pose、GT assignment、RMSD。它只能用于 label、诊断和上限分析。

## Plan of Work

第一步，修正 docking runner 文档语义。更新 `Docking/分子对接记录/full80_docking_exec_plan.md`，把容易误解的“成功”改为“流程跑通”或“Rosetta job 跑通”。保留 job 级失败和样本级 `ok` 的区别，明确该计划不计算真正的 docking 成功率。

第二步，新建 evaluation 子计划。文件放在 `Docking/分子对接记录/docking_evaluation_exec_plan.md`。它必须自包含地描述 GT ligand instance 表的来源、独立金属离子排除规则、前置 center hit、Hungarian precision/recall、RMSD 计算、重复 ligand occurrence 消歧、rank/top k/top k% 和分层统计。

第三步，实现 evaluation 主干。优先新增 `Docking/evaluation/` 子目录。第一版应包含一个 CLI 入口，读取一个或多个 docking run 的 `samples/*/audit/summary.json`、`results.json`、`assignments.json`、`postprocess.json`，读取 ligand mapping 和真实结构 ligand 坐标，输出 tables 到服务器允许目录中的 evaluation run 下。它要支持 partial runs：没有 summary 的 pending 样本跳过并记录。

第四步，新建 ML scoring 子计划。文件放在 `Docking/分子对接记录/docking_ml_scoring_exec_plan.md`。它必须说明 ML 只消费 evaluation 输出的 truth/evaluation/feature tables，不重新定义 GT。第一版同时实现少变量网格调参和 tree ensemble/ranking 路线；若服务器已有 LightGBM/XGBoost/CatBoost 则优先调用，若没有则用 sklearn RandomForest 或 HistGradientBoosting 做第一版。

第五步，实现 feature table 和 baseline scorer。特征表分 deployable 与 oracle 两类字段。deployable 字段包括 Rosetta dG/dH/lig_dens/ligscore、instance score_mean/score_max、voxel_count、pose 与预测 mask overlap、ligand heavy_atoms/半径/主轴比例、receptor 来源、map resolution、样本内 ligand/pred instance 统计。oracle 字段包括 GT id、GT center distance、GT pose RMSD、rank、top k、top k% 和 label。

第六步，同步和服务器运行。只能使用固定同步脚本。若 `268268` 仍在运行且占用 48 CPU，evaluation/feature job 最高再用 48 CPU。若 docking job 已完成，evaluation 可用最多 96 CPU。只取消 Codex 自己提交的任务，绝不能取消用户提交的任务。

第七步，持续更新总控和子计划。每个 milestone 完成时，总控计划记录状态，子计划记录具体发现、命令、输出表和失败归因。最终报告中必须把“流程跑通率”和“真实对接成功率”分开。

## Concrete Steps

在本地编辑和验证：

    cd C:\Users\15919\OneDrive\My_Project\Pocket_Plus
    python -m compileall Docking

同步服务器：

    C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat

服务器侧 evaluation 运行的预期形态如下，实际命令会在 evaluation 脚本完成后写入子计划：

    cd /home/penghongen/My_Project/Pocket_Plus
    source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
    python Docking/evaluation/run_evaluation.py --eval-run-id 20260520_eval_full80_partial --docking-run-id 20260519_full80_top1_nstruct2 --docking-run-id 20260520_full80_top1_pending13_scipy_required --jobs 48

## Validation and Acceptance

本地验收：

- `python -m compileall Docking` 通过。
- evaluation 模块可以在没有服务器数据的情况下导入。
- GT 构建、前置 center hit 和 RMSD 计算的核心函数有小型合成输入验证。

服务器验收：

- evaluation run 在 `/home/penghongen/分子对接尝试/evaluation_runs/{eval_run_id}` 下写出主表和 summary。
- summary 区分样本数、completed 样本数、pending 跳过样本数、流程跑通 Rosetta job 数、job 级失败数。
- 前置位点评估报告 3/4/8 Å hit，至少包含宽松多对一统计和 Hungarian precision/recall。
- 最终 pose 评估报告 RMSD <= 2/3/5 Å，且输出逐 pose RMSD 表。
- 真实匹配接近程度报告 rank、top k、top k%、正确 assignment cost 与当前最优 assignment cost 的比值或差值。
- 分层统计至少按样本、ligand CCD、ligand heavy_atoms 区间、是否糖类、是否含内部金属、receptor 来源、map resolution、前置 center hit 档位输出。
- ML feature table 明确标记 deployable/oracle 字段，训练脚本不能把 oracle 字段放入 deployable model。

## Idempotence and Recovery

所有服务器输出必须按 run id 隔离。重复运行 evaluation 应创建新的 `eval_run_id`，或在同一 `eval_run_id` 下覆盖自己生成的表，但不得写出 `/home/penghongen/分子对接尝试`。脚本应跳过 pending 样本并记录原因，不能因为一个样本缺少 summary 或一个 Rosetta job 失败而终止全局统计。

如果服务器任务卡住，只能取消 Codex 自己提交的 job。判断 job 所属时使用提交记录、run id 和 Slurm job 名称，不对用户提交的 job 执行 `scancel`。若无法确认 job 是否由 Codex 提交，默认不取消。

本地只修改 `Docking` 目录。不得回滚用户在其他目录的修改。若同步时发现远端代码与本地不同，只使用固定同步脚本，不临时发明同步路径。

## Artifacts and Notes

已知服务器输入：

    /home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy
    /storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv
    /storage/penghongen/EMDB_PDB_resolution_3.5.csv
    /storage/chenzhaoyang/cryo_em/CIF_3.5_atom
    /storage/chenzhaoyang/cryo_em/result_split
    /home/penghongen/software/rosetta

已知服务器输出根目录：

    /home/penghongen/分子对接尝试

当前必须避免的误读：

    824/844 Rosetta job 跑通

这句话不是 824 个“对接成功”。它只说明 824 个 Rosetta job 产生了可解析输出。真正成功率需要 evaluation 计算。

## Interfaces and Dependencies

总控计划预期维护下列接口：

- `Docking/分子对接记录/docking_master_exec_plan.md`：本文件，负责跨子任务状态和边界。
- `Docking/分子对接记录/full80_docking_exec_plan.md`：只负责 docking runner、流程跑通、solver、补跑和失败日志。
- `Docking/分子对接记录/docking_evaluation_exec_plan.md`：负责 GT、center hit、RMSD、rank/top k/top k% 和分层统计。
- `Docking/分子对接记录/docking_ml_scoring_exec_plan.md`：负责 feature table、网格调参、tree ensemble/ranking model 和 assignment-level objective。

evaluation 代码应优先使用 Python 标准库、NumPy、SciPy、joblib、RDKit 或 Biopython 中已经可用的能力。RMSD 原子对应优先使用 mol2 atom names 或原始 atom ids；若 atom correspondence 不可靠，输出 `rmsd_warning` 并从严格成功率中单独分层，而不是静默混入可靠 RMSD。

ML 代码可以检查并优先调用服务器已有的 LightGBM、XGBoost 或 CatBoost；若不可用，第一版使用 sklearn 的 RandomForest、HistGradientBoosting、LogisticRegression 或简单网格搜索。任何模型训练都必须记录输入特征列、排除的 oracle 列、训练样本数、验证方式和随机种子。

## Revision Notes

2026-05-20: 初始总控 ExecPlan 落盘。根据用户反馈，将本轮工作从“docking 流程跑通”提升为“真实几何评估与打分优化”，并明确资源边界、术语边界、evaluation 主干、ML scoring 双路线和 top k/top k% 评价维度。

2026-05-20: 记录用户对长时间前台命令的纠偏。新增 `Docking/sbatch/evaluation_cpu.sbatch`，后续长任务必须通过 sbatch/heartbeat 检查。
