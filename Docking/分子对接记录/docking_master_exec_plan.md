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
- [x] (2026-05-23) 发现旧 oracle array 每个 task 申请 8 CPU 但 Rosetta 实际基本串行单核运行；已建立逐 job 时间预算并把该问题记录为需修复的性能 bug。
- [x] (2026-05-23) 实现通用样本内部 Rosetta 并行执行层，oracle 与 realistic 均可通过 `--rosetta-jobs` 对所有独立 ligand-site-receptor jobs 并发运行；并发日志目录已隔离。
- [x] (2026-05-23) 首次 `7v19` 24 核覆盖 job `283182` 暴露旧 partial 参数文件不可直接覆盖与样本失败未传递 Slurm 退出码两项恢复问题；已修复批处理退出码逻辑。
- [x] (2026-05-23) 清理已授权覆盖的 `7v19` 单一样本旧输出后重提 24 核全量覆盖 job `283183`；该 job 已越过原参数准备冲突并在运行。
- [x] (2026-05-23) `283183` 的首轮监视已看到 wall `1:37` 对应累计 CPU `17:28`（约 `10.8x`），证明样本内部并行实际生效。
- [x] (2026-05-23) `283183` 在 wall `6:26` 对应累计 CPU `2:12:50`（约 `20.6x`），无 stderr 或样本错误且节点内存余量充足；据此取消旧 `7ut7` 串行 task 与旧 collector，提交 16 核覆盖 job `283281` 与最终 easy20 collector `283282`。
- [x] (2026-05-23) 输出层验收完成：`283183` 已写出首批 24 个 scorefile 且约 `21.7x` CPU/墙钟，`283281` 启动后约 `9.5x` CPU/墙钟；样本内部并行可作为后续 oracle/realistic 调度主线。
- [x] (2026-05-24 12:02+08:00) oracle 长尾覆盖与最终汇总结束：`283183`（`7v19`, 24 CPU）完成于 `10:29:41`，`283281`（`7ut7`, 16 CPU）完成于 `04:50:59`，`283282` 已生成完整 easy20 `batch_summary.json`。
- [x] (2026-05-24 12:02+08:00) 读取完整 oracle 汇总：20/20 样本有 summary，5800 个 Rosetta jobs 中 5739 个流程跑通；61 个失败全部集中于 `6bk8`。400 个 assignment 均记录 `solver=dp_virtual`；当前表尚未计算 RMSD。
- [x] (2026-05-24 12:02+08:00) 检查 realistic `281985/281986`：12/20 样本完整完成且 636/636 jobs 流程跑通，8 个 1-CPU 串行长尾仍运行；按串行 P75/P90，决定收尾的尾部仍约需 186/220 小时。
- [x] (2026-05-24 13:05+08:00) 用户授权增量严格评价、realistic 尾部同 run 覆盖，以及 `6bk8` 若能短时修复则在原 oracle run 下覆盖重跑并自动刷新最终质量指标。
- [x] (2026-05-24 13:18+08:00) 新增 oracle 专用严格 evaluator 与 sbatch 入口；修复 receptor 转换对 `UNK` polymer residue 的处理；新增 realistic 单样本内部并行 sbatch 入口并使用固定同步脚本同步。
- [x] (2026-05-24 13:25+08:00) 当前 oracle 初始严格评价 job `284650` 完成：严格选中 pose RMSD `<=2/3/5 Å` 为 `5/67/319`（分母 2240），Hungarian occurrence 级匹配为 `562/1120`；该结果含修复前 `6bk8`，用于即时诊断而非最终表。
- [x] (2026-05-24 13:25+08:00) `6bk8` 修复 smoke job `284651` 完成：过滤 `UNK` 后原稳定失败的 `true_center_identity + true_receptor` 路径为 `2/2` jobs 流程跑通；已提交权威覆盖链 `284663 -> 284664 -> 284665`。
- [x] (2026-05-24 13:24+08:00) realistic 已完成的 `6bk8` 与接近完成的 `8vcj` 保持不动；仅覆盖 6 个仍明显长尾样本，提交 `284652..284657` 与新 collector `284658`，并观测到新任务确实使用样本内多核。
- [x] (2026-05-24 13:27+08:00) heartbeat 复查运行健康度：`284663` 在墙钟 `00:06:38` 时累计 CPU `00:56:12` 且 stderr 为空；六个 realistic 覆盖任务与保留的 `8vcj` 均仍运行、覆盖任务 stderr 为空，realistic 当前已有 `13/20` 个样本 summary。
- [ ] (2026-05-24) 等待 `284663/284664/284665` 与 `284652..284658` 反馈；将最终 oracle 严格结果写入 `oracle_easy20_readable.md`，并对 realistic 完整产物运行同口径评价。
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

- Observation: CPU 资源申请必须与 runner 的真实并行层绑定，不能仅靠 Slurm 申请更多核数。
  Evidence: 旧 oracle task 的 `TotalCPU` 约等于 wall time，而非申请的 8 倍；新实现新增 `run_rosetta_jobs(..., rosetta_jobs)` 并让 oracle/realistic 入口显式传入并行数。

- Observation: 长尾覆盖重跑需要显式处理 partial 输出幂等性和批处理退出码。
  Evidence: `7v19` 验证 job `283182` 因已有 params 文件在准备阶段失败，stdout 已写 `num_failed=1` 但 Slurm 仍显示 `COMPLETED ExitCode=0:0`；两个 batch 入口现已改为失败摘要写完后返回非零。

- Observation: 公共样本内部并行执行层已从代码承诺变为真实服务器证据。
  Evidence: 清理并重提的 `7v19` job `283183` 在 wall time `1:37` 时已有累计 CPU `17:28`，而旧 8 核任务累计 CPU 约等于 wall time；说明 Rosetta 子进程并发确实工作。

- Observation: 样本内部并行已经使完整 `easy20` oracle 能够收尾，但该完成仅表示产物完整与 Rosetta job 大多跑通。
  Evidence: `283183` 完成于 `10:29:41`，`283281` 完成于 `04:50:59`，collector `283282` 完成；最终 `batch_summary.json` 为 `num_samples=20`、`num_jobs=5800`、`num_success=5739`、`num_failed=0`、`num_skipped=0`。该 summary 不含 pose RMSD 指标。

- Observation: 完整 oracle 的 Rosetta job 失败高度集中，当前首先应作为结构兼容性缺陷排查，而非总体 docking 结论。
  Evidence: 61 个失败均属于 `6bk8`，其中 true receptor 为 60/2900 个失败、cryoatom receptor 为 1/2900 个失败；代表性 stderr 报 `Stub::from_four_points()` 无法从重合点构造向量，继而触发 zero-length normalized vector 内部错误。

- Observation: `runtime_budget_v1` 仍是并行改造前 18 个完整样本的串行单核基线，不应直接用新并行覆盖产物覆写为同一统计总体。
  Evidence: 当前表记录 `completed_sample_count=18`、成功 job P75/P90 为 `331.1/390.6 s`；最终新增的 `7ut7` 与 `7v19` 是 16/24 CPU 样本内并行输出，其单 job wall time 与串行基线不是同一执行条件。

- Observation: realistic 的 1-CPU 修复消除了空申请 CPU，却没有消除样本级串行长尾。
  Evidence: `281985` 已完成 12/20 样本且 636/636 jobs 流程跑通；仍运行的 `8ca3`、`8xh9`、`7z7s` 分别剩余约 2027、1999、1804 个串行 Rosetta jobs，按串行 P75/P90 预算，最慢尾部仍需约 `186/220` 小时。

- Observation: `6bk8` 的主失败机制已经从怀疑归因变为可复现且可修的 receptor 转换缺陷。
  Evidence: 旧 `true_receptor` PDB 中含 `1175` 个 `UNK` 原子，`cryoatom_receptor` 不含 `UNK`；61 个失败中 60 个发生在 `true_receptor`，stderr 明确报 `No mainchain score table for residue type pdb_UNK exists`。过滤 `UNK` 后 smoke job `284651` 的原失败路径 `2/2` 跑通。

- Observation: oracle 初始严格评价已证明“给真实中心”仍不足以产生高质量 pose。
  Evidence: `284650` 读取修复前 oracle 完整表并输出：真实 pair 已运行时可靠 RMSD `<=2/3/5 Å` 为 `5/73/449`（可靠分母 2200）；严格选中后 `<=2/3/5 Å` 为 `5/67/319`（全体分母 2240）；Hungarian exact occurrence accuracy 为 `562/1120 = 50.18%`。

- Observation: realistic 尾部迁移后，多核能力在真实流程上也获得了立即证据。
  Evidence: `284653` 在墙钟约 `5:10` 时累计 CPU `38:09`，`284654` 为 `45:58`，明显高于串行 `1x`；迁移仅覆盖 `8bly, 8ut3, 8wis, 8ca3, 7z7s, 8xh9` 六个未完成长尾样本。

- Observation: 修复覆盖链和 realistic 迁移在首次 heartbeat 时保持健康运行，尚未出现需要改变实验定义的新故障。
  Evidence: `284663` 墙钟 `00:06:38`、累计 CPU `00:56:12`，约 `8.5x` 活跃 CPU/墙钟且 stderr 为 0 字节；`284652..284657` 墙钟均为 `00:12:50`，其中 `284653/284654` 累计 CPU 已为 `01:42:00/01:58:02`，各 stderr 均为空。realistic 已存在 13 个完成样本 summary，collector 仍正确等待剩余任务。

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

- Decision: 样本内部 Rosetta 并行是 runner 公共能力，不按 identity/Hungarian 或 oracle/realistic 各自实现；CPU 推荐以 `ceil(计划 jobs / 40)` 为简单口径，资源宽裕时可参考 `/30`，且总申请不超过 96 核。
  Rationale: 每个 Rosetta 子进程都对应独立 ligand-site-receptor docking job；公共并行层让后续实验可以复用同一种资源估算与调度方法，同时避免为每个实验类型维护不同性能实现。
  Date/Author: 2026-05-23 / 用户与 Codex

- Decision: 暂不把完整 oracle 的并行覆盖结果混入 `runtime_budget_v1` 的串行单 job 分位数；后续时间预算必须显式区分串行基线与样本内并行策略。
  Rationale: 把不同并行策略下的 per-job wall time 合成一个 P75/P90，会使 AI agent 无法可靠回答“给定 CPU 配置需要多久”。当前串行表仍可用于估算未迁移的 realistic 尾部，并行任务则应记录样本 wall time、分配核数与实际加速。
  Date/Author: 2026-05-24 / Codex

- Decision: oracle 严格 evaluation 独立实现并显式分离 `truth_pair_pose` 与 `strict_selected_pose`；最终用户结论文档只记录 `6bk8` 修复覆盖后的最终版本。
  Rationale: oracle 的 task/variant 结构自带真实 site occurrence，旧 realistic evaluator 的 label-only rank 口径会高估匹配质量；同时用户指定 `oracle_easy20_readable.md` 只用于最终任务定义、配置、数据和结论。
  Date/Author: 2026-05-24 / 用户与 Codex

- Decision: realistic 尾部不全量撤销正在跑的结果，只替换仍显著长尾的六个样本，并为 oracle `6bk8` 24 CPU 覆盖预留资源。
  Rationale: 状态检查时 realistic 的 `6bk8` 已完成，`8vcj` 仅剩很短尾部；覆盖它们只会丢掉有效进展。六个替换任务合计申请 65 CPU，叠加保留任务和 oracle 覆盖仍不超过 96 CPU。
  Date/Author: 2026-05-24 / Codex

## Outcomes & Retrospective

当前处于总控计划落盘阶段。下一步要把已有 docking 计划的术语修正为“流程跑通”，并新建 evaluation 与 ML scoring 子计划。真正的阶段性成果应是：在 `268268` 全部完成之前，已经能对已有样本输出前置 center hit、RMSD、rank/top k/top k% 和分层统计表。

2026-05-20 更新：总控计划、evaluation 子计划、ML scoring 子计划已经落盘。`Docking/evaluation/run_evaluation.py` 第一版已经实现并通过本地 `python -m compileall Docking` 与 `Pocket_Plus_windows` 环境下的 `--help` 验证。下一步是同步服务器并在允许目录启动 partial evaluation run。

2026-05-20 再更新：已同步服务器并完成 `20260520_eval_full80_partial_v1`。这次结果只应视为第一版诊断信号，不是最终成功率：RMSD 用 direct atom-order 方法，rank/top k 尚未约束 GT occurrence。关键收获是前置 center hit 和最终 RMSD 都明显偏低，说明 evaluation 方向必须优先推进。

2026-05-21 更新：完成一次本地口径复核和文档推进。`best_summary.json` 的 `avg_instance_*` 不应再被称为 center hit；已新增 `docking_metrics_readable.md` 定义所有关键指标的分母、分子、阈值和 caveat。随后用用户补充的密码完成服务器只读检查：`268268` 仍在运行，补跑 run 已完成 10/13 个样本，合并后 full80 top1 当前 77/80 样本有 summary、已完成样本 Rosetta job 1886/1950 跑通。下一步应等 `268268` 收尾或下一次 heartbeat 继续检查，而不是现在提交新的重型任务。

2026-05-21 00:55+08:00 更新：本轮服务器 SSH 被本地沙箱 socket 权限拦截，因此 `268268` 状态没有新确认。为避免空转，已在本地实现前置 center Hungarian precision/recall；下一次服务器可访问时，应同步 Docking 后重跑 evaluation，把 loose hit 与 Hungarian precision/recall 同时写入 summary。

2026-05-23 更新：easy20 实验暴露了调度性能缺陷：旧 array 为样本申请多核但内部 Rosetta 仍串行，长尾因此被人为放大。现已把样本内所有独立 Rosetta jobs 抽象为公共并行队列，并将 `7v19` 从“准备丢弃的长尾”改为“24 核真实性能验证及 easy20 恢复入口”。若观测证实并行生效，后续所有新 oracle/realistic 实验都将按 job 数建议分配 `rosetta_jobs`，而评估指标仍严格与流程跑通分离。

2026-05-23 21:47+08:00 再更新：`7v19` 的首次验证提交先暴露了恢复路径缺陷，而不是性能结论：原 partial 参数文件导致全量覆盖在 Rosetta 运行前停止，并且样本失败曾未反映为 Slurm 失败。现已修复失败退出码，接下来仅清空用户授权覆盖的 `7v19` 输出再重提，继续获得并行是否有效的直接反馈。

2026-05-23 21:52+08:00 再更新：新的 `7v19` 覆盖 job `283183` 已在清空该单一样本输出后启动并越过参数准备阶段；目前仍是执行中反馈，下一判断点是确认它在 24 核分配下真正并发运行 Rosetta jobs，再决定是否对 `7ut7` 采用同一路线。

2026-05-23 21:56+08:00 再更新：`283183` 已提供约 `20.6x` CPU/墙钟并行证据且未出现新的错误或内存压力，因此并行执行层已进入主线。旧 `7ut7` 串行 task 与会过早汇总的 collector 已取消，新的 `7ut7` 16 核覆盖 job `283281` 与最终 easy20 汇总 `283282` 已提交。下一阶段不再围绕“是否剔除长尾样本”讨论，而是等待完整结果后用真实 evaluation 衡量 docking 质量。

2026-05-23 22:02+08:00 再更新：`7v19` 已产生首批正常 scorefile，排除了“只有 CPU 活跃但输出路径损坏”的担忧；`7ut7` 同一执行模式也已实际使用多核。当前 oracle 覆盖完成时间由 `7v19` 主导，保守预计从其提交起约 `6-10` 小时完成，随后自动汇总 full easy20。该结果只证明性能与流程写出改善，仍必须等待 assignment/RMSD evaluation 才能评价科学质量。

2026-05-24 12:02+08:00 更新：样本内部并行已让 oracle 恢复完整 easy20 并收尾。`7v19` 的 24 CPU 覆盖实际用时 `10:29:41`，`7ut7` 的 16 CPU 覆盖实际用时 `04:50:59`，最终 collector 已输出 20 样本、5800 jobs 的总表；其中 5739 jobs 流程跑通（98.95%），61 个失败全部来自 `6bk8` 的 Rosetta 几何内部错误。当前 400 个 assignment 仅确认使用 `dp_virtual`，尚无 assignment correctness 或 RMSD，因此不能称为真正对接成功。realistic 新 run 仍有 8 个串行尾部，按现有 P75/P90 预算最慢仍约需 `186/220` 小时；下一步应优先决定是否用已经验证有效的样本内并行替换这些尾部，并为 oracle 输出启动严格 evaluation。

2026-05-24 13:25+08:00 更新：本轮已经从“等待结果”转为“用结果推动修复”。`6bk8` 的 true receptor 中含 Rosetta 无法打分的 `UNK` polymer residue，过滤后最小 smoke `284651` 成功，因此已启动 24 CPU 权威覆盖并串接最终 collector/evaluation。与此同时，新的 oracle 严格 evaluator 已在修复前结果上给出即时诊断：严格 RMSD `<=2 Å` 仅 `0.22%`、`<=5 Å` 为 `14.24%`，Hungarian exact occurrence accuracy 为 `50.18%`，这表明下游 pose 和 scoring/matching 都需继续优化。realistic 的六个真正长尾样本已切换到多核覆盖并获得 CPU 利用证据；最终 oracle 数据产出后将写入用户指定的 `oracle_easy20_readable.md`。

2026-05-24 13:27+08:00 更新：首次自动复查确认这两条执行线没有在迁移后立刻失效。Oracle 的 `6bk8` 覆盖已表现出多核实际占用且无 stderr，最终 collector/evaluation 仍在依赖队列；realistic 当前 `13/20` 个样本已有 summary，六个覆盖任务均持续多核运行且无错误流。当前仍应等待最终严格指标与完整 realistic 汇总，而不是把健康运行直接解释为 pose 成功。

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

2026-05-23 调度修订：后续运行先计算样本的计划 Rosetta job 数。普通短样本可以使用 1 核 array；需要缩短长尾时，样本内并行通过 `--rosetta-jobs` 显式启用，推荐 CPU 为 `ceil(jobs / 40)`，资源宽裕时可用 `ceil(jobs / 30)` 作更积极预算。该推荐不替代服务器总计不超过 96 核与仅取消 Codex-owned job 的硬边界。

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
