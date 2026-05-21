# 分子对接真实评估 ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

本计划是 `Docking/分子对接记录/docking_master_exec_plan.md` 的 evaluation 子计划。它只负责真实评估：构建 GT ligand instance、评价前置预测位点、计算最终 docked pose RMSD、统计真实匹配在当前打分下的 rank/top k/top k%，并输出样本级、位点级、ligand 级和分层报表。它不负责提交 Rosetta docking，也不负责训练 ML 模型。

## Purpose / Big Picture

完成本计划后，用户可以把已有 docking run 的审计输出转化为真正的评价结果。评价必须回答四个问题：预测 instance center 是否靠近真实 ligand center；Hungarian 匹配后 ligand identity 和 occurrence 是否正确；docked pose 与真实 pose 的 RMSD 是否达到 2/3/5 Å；如果当前最优匹配错了，真实匹配在候选排序里到底离最优有多近。

## Progress

- [x] (2026-05-20 13:35Z) 用户确认 GT pose 以 ligand mapping 中的 native mol2 / 真实结构 ligand 坐标为准。
- [x] (2026-05-20 13:35Z) 用户确认前置位点使用 instance center 与真实 ligand center 的距离评估，阈值至少包含 3/4/8 Å。
- [x] (2026-05-20 13:35Z) 用户确认最终严格标准以 RMSD <= 2.0 Å 为主，同时报告 2.0/3.0/5.0 Å 三档。
- [x] (2026-05-20 13:35Z) 用户确认必须加入真实匹配接近程度，包括 rank、top k、top k% 和正确 assignment cost 相对当前最优 cost 的关系。
- [x] (2026-05-20 13:50Z) 实现 `Docking/evaluation/run_evaluation.py` 的第一版 CLI，能读取 completed/partial docking audit 并写出基础 evaluation tables。
- [x] (2026-05-20 13:50Z) 在本地完成 `python -m compileall Docking` 和 `Pocket_Plus_windows` 环境下 `run_evaluation.py --help` 验证。
- [x] (2026-05-20 14:05Z) 同步到服务器并完成 `20260520_eval_smoke3` 小样本 smoke，确认 evaluation CLI 能在真实服务器路径上运行。
- [x] (2026-05-20 14:15Z) 完成 `20260520_eval_full80_partial_v1`，在已有 partial full80 输出上写出第一版 evaluation tables。
- [x] (2026-05-20 14:20Z) 将 evaluation 结果摘要回写本计划和总控计划。
- [x] (2026-05-21 00:34+08:00) 本地核对 `two_stage_basic.py`、`voxel_tuning.py` 与 `voxel_evaluator.py`，确认 `best_summary.json` 的 `avg_instance_*` 是体素连通域覆盖口径，不是 docking 前置 center hit。
- [x] (2026-05-21 00:34+08:00) 新增用户可读指标解释文档 `Docking/分子对接记录/docking_metrics_readable.md`，逐项定义分母、分子、阈值和 caveat。
- [x] (2026-05-21 00:55+08:00) 在 `Docking/evaluation/run_evaluation.py` 中实现前置 center Hungarian precision/recall，并保留 loose 多对一 hit 旧口径。
- [x] (2026-05-21 00:55+08:00) 本地验证 `python -m compileall Docking` 通过；用 `Pocket_Plus_windows` 环境运行合成样例，确认 2 个 GT、3 个 pred 时 Hungarian recall 为 1.0、precision 为 2/3。
- [ ] 修正 rank/top k 指标：必须同时约束 GT occurrence 与 pred site，不再只看 ligand label 是否属于 GT label 集合。
- [ ] 在服务器可访问后重跑 full evaluation，把前置 center Hungarian precision/recall 写入远端 evaluation summary。
- [ ] 增强 RMSD atom correspondence，优先用 atom name / RDKit 对齐，并对糖类、对称 ligand、含金属 ligand 单独分层。

## Surprises & Discoveries

- Observation: 早期 `progress1_readable.md` 已明确指出当前还没有计算 RMSD 或 identity accuracy。
  Evidence: 该文件第 360 行附近的风险列表写有“还没有计算 RMSD 或 identity accuracy”。

- Observation: 第一版 evaluation CLI 已能在真实 partial full80 数据上跑通，但当前 rank/top k 指标会高估。
  Evidence: `20260520_eval_full80_partial_v1` 输出 75 个样本、828 个 truth instances、66 个 selected pred sites、1642 个 pose rows；rank_summary 的 `truth_top1_rate=1.0` 明显不合理，原因是第一版只按 ligand label 是否属于 GT 集合判定 truth，没有绑定 GT occurrence 和 pred site。

- Observation: direct atom-order RMSD 第一版已经给出强烈负面信号，但不能作为最终严格结论。
  Evidence: `20260520_eval_full80_partial_v1` 中 1618 个有效 direct RMSD 里，RMSD <= 2 Å 为 0，<= 3 Å 为 0.31%，<= 5 Å 为 1.17%。由于 atom correspondence 尚未增强，该结果是优先排查方向，不是最终论文式成功率。

- Observation: `best_summary.json` 的 `avg_instance_f1` 不能解释为前置 center hit。
  Evidence: `two_stage_basic.py` 调用 `run_voxel_param_search` 生成 `stage2_threshold_component_policy/best_summary.json`；`voxel_tuning.py` 的 summary 来自 `evaluate_instance_mask`；`voxel_evaluator.py` 中 instance precision/recall 的分子分别是满足 `pred_cover_ratio >= alpha` 的预测连通域数和满足 `gt_cover_ratio >= beta` 的 GT 连通域数，而不是 center distance <= 3/4/8 Å。

- Observation: 远端 `best_summary.json` 已确认存在，但它不改变 evaluation 计划的优先级。
  Evidence: 密码登录服务器后读取到 `best_summary.json`：`avg_instance_precision=0.4690510621159653`、`avg_instance_recall=0.47170958421751286`、`avg_instance_f1=0.36896760343806495`、`avg_num_candidates=28.3375`。这些仍是 voxel/instance 覆盖指标，不能替代 `Docking/evaluation/run_evaluation.py` 的 center hit、RMSD 与 rank/top-k。

- Observation: 本地 evaluation 已具备前置 center 的一一匹配统计，但远端 full evaluation 尚未重跑。
  Evidence: `evaluate_site_hits` 现在输出 `hungarian_hit_le_3A/4A/8A` 明细，以及 `hungarian_recall_le_*`、`hungarian_precision_le_*` summary；本轮 SSH 被本地 socket 权限拦截，未能在 `/home/penghongen/分子对接尝试/evaluation_runs` 新建或刷新 evaluation run。

## Decision Log

- Decision: 前置位点评估分成宽松多对一统计与 Hungarian precision/recall 两套。
  Rationale: 宽松统计能回答“模型有没有在附近预测出点”，Hungarian 统计能回答“预测点和真实点作为集合的一一匹配质量如何”。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 重复 ligand 的 occurrence 消歧必须再做一层 Hungarian，而不是只看 CCD 是否相同。
  Rationale: 多个 `NAG`、`ATP` 或糖链片段在同一样本中可能同时存在，只看 CCD 会高估 identity accuracy。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: RMSD 结果必须携带 `rmsd_method` 和 `rmsd_warning`。
  Rationale: atom correspondence 对糖类、对称 ligand、含金属 ligand 可能不稳定，不能把不可靠 RMSD 静默混入严格成功率。
  Date/Author: 2026-05-20 / Codex

- Decision: 后续 full evaluation、feature table 和 ML 训练必须通过 sbatch 运行，短 smoke 才能使用前台 SSH。
  Rationale: 用户指出长时间前台命令会造成无进展等待；服务器任务应通过 Slurm 日志、lock 和 heartbeat 可恢复地运行。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 指标汇报中禁止把 `avg_instance_*` 命名为 center hit；前置 center hit 只保留给 evaluation 中基于预测 center 与 GT center 距离的 3/4/8 Å 指标。
  Rationale: 连通域覆盖和中心距离回答的是不同问题；混用会高估或误读前置预测对 docking 的有效性。
  Date/Author: 2026-05-21 / Codex

- Decision: `site_hit_summary` 同时保留 loose hit 与 Hungarian precision/recall。
  Rationale: loose hit 适合判断 GT 附近是否有预测点，Hungarian precision/recall 适合判断预测集合的一一匹配质量；两者分母不同，不能互相替代。
  Date/Author: 2026-05-21 / Codex

## Outcomes & Retrospective

2026-05-20 更新：第一版 evaluation CLI 已经实现。它目前可以读取 docking run audit、构建 GT center 表、提取 selected pred sites、提取 pose/result/assignment 表、计算前置 center hit、做 direct atom-order RMSD 第一版、计算 assignment pair rank/top k 接近程度。下一步需要同步到服务器，在真实 partial run 上运行并根据实际输出修正 RMSD 解析和 occurrence 消歧。

2026-05-20 再更新：服务器 partial evaluation 已完成。前置 center hit 和 direct RMSD 都很低，证明必须尽快从“流程跑通”转向真实评价。但 rank/top k 需要立即修正 truth 定义，否则会产生虚高的 truth_top1。

2026-05-21 更新：本地口径核对确认 `best_summary.json` 的 `avg_instance_*` 属于推理后处理 mask/instance 覆盖指标，不是 center hit。已经新增 `docking_metrics_readable.md` 定义指标口径。服务器只读检查随后确认 `268268` 仍在运行；因此本轮不启动新的 full evaluation sbatch，避免和 48 CPU docking job 叠加超出资源预算。下一步 evaluation 重点仍是修正 rank/top-k 的 occurrence 约束，并在 `268268` 收尾后重跑 full evaluation。

2026-05-21 00:55+08:00 更新：本地完成前置 center Hungarian precision/recall 实现。当前仍不能把 rank/top-k 当作 assignment 成功率；下一步在服务器可访问后同步 Docking 并重跑 evaluation，优先比较 loose center hit 与 Hungarian precision/recall 的差距。

## Context and Orientation

本地只允许编辑 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking`。服务器只允许写 `/home/penghongen/分子对接尝试`。已有 docking run 输出位于服务器：

    /home/penghongen/分子对接尝试/pipeline_runs/20260519_full80_top1_nstruct2
    /home/penghongen/分子对接尝试/pipeline_runs/20260520_full80_top1_pending13_scipy_required

GT ligand 相关输入来自：

    /storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv
    /storage/chenzhaoyang/cryo_em/CIF_3.5_atom
    /storage/penghongen/EMDB_PDB_resolution_3.5.csv

`流程跑通` 是 Rosetta job 产生可解析输出。`真实对接成功` 是 evaluation 基于 GT、RMSD 和 match accuracy 得到的指标。

## Plan of Work

第一步，新增 `Docking/evaluation/run_evaluation.py`。该 CLI 读取一个或多个 `--docking-run-id`，输出到 `/home/penghongen/分子对接尝试/evaluation_runs/{eval_run_id}`。第一版必须能跳过 pending 样本，并记录 skipped reason。此步骤已完成第一版。

第二步，构建 GT ligand instance 表。读取 ligand mapping 中 `PASS_HIGH` 且 `FORMAT_OK_FOR_LIBRARY_ENTRY` 的 mol2，排除独立金属离子，保留内部含金属 ligand 的标记。每个 GT 条目至少包含 `sample_id`、`ccd_id`、`label`、`mol2_path`、`heavy_atoms`、`internal_metals`、`center_xyz` 和可用于 RMSD 的 atom table。

第三步，构建 predicted site 表。读取每个样本的 `postprocess.json` 与 `summary.json`，记录 raw/output/selected site、instance score、voxel count、center 和是否被用于 docking。

第四步，构建 pose/result 表。读取 `results.json` 中每个 Rosetta job 的 best score row、decoy summary、stdout/stderr、scorefile 和输出 PDB。对有输出 PDB 的 pose，解析 ligand 坐标并与 GT ligand occurrence 计算 RMSD。

第五步，计算前置位点指标。对 pred center 与 GT center 计算距离矩阵，输出 3/4/8 Å 宽松命中、Hungarian precision/recall、每个 GT 的最近 pred、每个 pred 的最近 GT。

第六步，计算最终 pose 指标。按 sample/site/ligand/receptor/GT occurrence 消歧，输出 RMSD <= 2/3/5 Å、最佳 RMSD、按 CCD 与 occurrence 的正确率。

第七步，计算真实匹配接近程度。对当前 cost 或 score 排序，输出正确 pair 的 rank、top k、top k%。对整样本 assignment，输出正确 assignment cost、当前最优 assignment cost、两者差值和比值；若 cost 为负或零，保留差值并对比值加 warning。

第八步，输出分层统计。至少按样本、CCD、heavy atom 区间、是否糖类、是否含内部金属、receptor 来源、map resolution、前置 center hit 档位、ligand shape 相似度区间统计。

## Concrete Steps

本地验证：

    cd C:\Users\15919\OneDrive\My_Project\Pocket_Plus
    python -m compileall Docking

服务器运行预期：

    cd /home/penghongen/My_Project/Pocket_Plus
    source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
    python Docking/evaluation/run_evaluation.py --eval-run-id 20260520_eval_full80_partial --docking-run-id 20260519_full80_top1_nstruct2 --docking-run-id 20260520_full80_top1_pending13_scipy_required --jobs 48

## Validation and Acceptance

验收标准：

- 输出 `truth_instances.csv`、`pred_sites.csv`、`pose_results.csv`、`site_hit_metrics.csv`、`pose_rmsd_metrics.csv`、`rank_metrics.csv`、`summary.json`。
- `summary.json` 明确区分 processed samples、pending samples、sample-level errors、Rosetta job 跑通率和真实评价指标。
- 任一 pending 样本不会让全局 evaluation 失败。
- RMSD 不可靠的条目必须有 warning，并在 summary 中单独计数。

## Idempotence and Recovery

重复运行同一 `eval_run_id` 只覆盖该 evaluation run 自己生成的表。任何服务器写入都必须留在 `/home/penghongen/分子对接尝试/evaluation_runs/{eval_run_id}` 下。若遇到缺失文件，记录到 `skipped_samples.csv` 或 `warnings.json`，继续处理其他样本。

## Artifacts and Notes

第一版不追求完美 RMSD 对称性处理，但必须透明标记 `rmsd_method`。如果 RDKit 可用，优先使用 RDKit 解析 mol2 和 PDB ligand；如果不可用，先用轻量 PDB/mol2 atom-name 对齐实现，并把方法记录为 `atom_name_direct`。

## Interfaces and Dependencies

预期 CLI：

    python Docking/evaluation/run_evaluation.py --eval-run-id <id> --docking-run-id <run_id> [--docking-run-id <run_id2>] [--jobs N]

预期核心函数：

- `load_truth_instances(mapping_csv, sample_ids) -> list[dict]`
- `load_docking_audits(run_roots) -> dict`
- `evaluate_site_hits(truth, pred_sites, thresholds) -> dict`
- `evaluate_pose_rmsd(truth, results) -> dict`
- `compute_rank_metrics(assignments, truth) -> dict`
- `write_evaluation_tables(output_dir, tables) -> None`

## Revision Notes

2026-05-20: 初始 evaluation 子计划落盘，响应用户关于前置位点、RMSD、top k/top k% 和分层统计的要求。

2026-05-20: 完成第一版 `Docking/evaluation/run_evaluation.py`。本地验证通过，但尚未在服务器真实数据上运行；RMSD 当前为 direct atom-order 方法，后续需要根据真实 PDB 输出检查 ligand atom 解析和 occurrence 消歧。

2026-05-20: `20260520_eval_smoke3` 和 `20260520_eval_full80_partial_v1` 已在服务器允许目录完成。记录下一步：修正 rank/top k 的 GT occurrence 约束，增强 RMSD atom correspondence，并把长任务迁移到 `Docking/sbatch/evaluation_cpu.sbatch`。
