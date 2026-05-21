# 分子对接匹配打分与机器学习 ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

本计划是 `Docking/分子对接记录/docking_master_exec_plan.md` 的 ML scoring 子计划。它只消费 evaluation 计划产出的 truth/evaluation/feature tables，不重新定义 GT。目标是改进 Hungarian 匹配前的 cost matrix，使正确 ligand、正确 site assignment 和最终低 RMSD pose 更容易被选中。

## Purpose / Big Picture

完成本计划后，用户可以比较两类打分方案：少变量网格调参的透明 baseline，以及 tree ensemble 或 ranking model 的强模型方案。模型输出不是最终科学结论，而是用于构造 Hungarian cost matrix 的候选打分函数。最终好坏必须用 evaluation 计划定义的样本级 assignment accuracy、RMSD <= 2/3/5 Å、rank/top k/top k% 来验证。

## Progress

- [x] (2026-05-20 13:40Z) 用户确认 ML scoring 应同时尝试同一预测 site 内 ligand 排名和整样本 Hungarian assignment 后总体正确率。
- [x] (2026-05-20 13:40Z) 用户确认 assignment-level objective 可以连续化，考虑正确率以及正确 assignment cost 相对当前最优 cost 的关系。
- [x] (2026-05-20 13:40Z) 用户确认强模型路线可以直接上 tree ensemble 或 ranking model，同时保留少变量网格调参。
- [x] (2026-05-20 13:40Z) 用户确认 `gt_instance` 不进入可部署模型特征，只用于 label、objective、分层统计和 oracle analysis。
- [ ] 等 evaluation 第一版产出 feature/evaluation tables。
- [ ] 实现少变量网格调参 baseline。
- [ ] 检查服务器可用 ML 包，优先尝试 LightGBM/XGBoost/CatBoost；不可用时使用 sklearn RandomForest 或 HistGradientBoosting。
- [ ] 实现 tree ensemble/ranking 第一版训练与验证。
- [ ] 把模型分数回填 cost matrix，运行 Hungarian 并比较 assignment-level 指标。
- [ ] (2026-05-21 00:55+08:00) 本轮因本地沙箱阻止 SSH socket，未能检查远端 `/home/penghongen/分子对接尝试/model_runs` 是否出现新反馈。

## Surprises & Discoveries

- Observation: `progress1_readable.md` 后半部分已经给出比简单二分类更贴近任务的 ML 方向。
  Evidence: 该文档建议对每个 site-ligand pair 提取 Rosetta score、instance score、mask overlap、ligand 属性、receptor 来源和样本内统计，并把目标设置为正确 ligand 排名前、正确 assignment 成本最小或正确率提升。

## Decision Log

- Decision: 第一版 ML 不只做 binary classification，而是同时保留 pair label、within-site ranking 和 assignment-level validation。
  Rationale: Hungarian 前的 cost matrix 关心候选之间的相对顺序和整样本组合，而不是孤立 pair 的概率校准。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: 强模型路线可以优先于简单模型启动，但必须保留少变量网格 baseline。
  Rationale: tree ensemble 可能更快捕获非线性交互，网格 baseline 则提供可解释对照。
  Date/Author: 2026-05-20 / 用户与 Codex

- Decision: deployable 特征和 oracle 字段必须在表中分栏或显式标记。
  Rationale: 防止训练脚本误把真实 GT 信息作为推理时可见特征使用。
  Date/Author: 2026-05-20 / 用户与 Codex

## Outcomes & Retrospective

当前尚未训练模型。本计划等待 evaluation 第一版产出可靠 feature/evaluation tables 后开始实现训练脚本。

2026-05-21 00:55+08:00 更新：本轮没有服务器 model_runs 新反馈，因为 SSH socket 被本地沙箱阻止。evaluation 已在本地补充前置 center Hungarian precision/recall；ML 仍应等待 rank/top-k occurrence 绑定和 feature/evaluation tables 更可靠后再启动。

## Context and Orientation

当前 docking pipeline 的手工 cost 大致来自 Rosetta 分数和 shape proxy。早期文档指出这只适合探索，缺少 docking 后 pose 与预测 mask 的空间重叠、ligand 拓扑、局部密度、样本内统计和 receptor 来源等特征。

ML scoring 的输入必须来自 evaluation 输出表，推荐位置为：

    /home/penghongen/分子对接尝试/evaluation_runs/{eval_run_id}

本地代码放在：

    C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking\ml_scoring

## Plan of Work

第一步，定义 feature schema。字段分成 deployable、oracle、label、group 四类。deployable 字段包括 Rosetta dG、dH、lig_dens、ligscore、total_score、instance score_mean、score_max、voxel_count、ligand heavy_atoms、ligand 半径、是否糖类、是否含内部金属、receptor 来源、map resolution、样本内 ligand/pred instance 数量统计、pose 与预测 mask overlap。oracle 字段包括 GT id、GT center distance、RMSD、正确 assignment、rank 和 top k/top k%。label 字段包括 pair 是否正确、RMSD <= 2/3/5 Å、site 内正确 ligand rank。

第二步，实现少变量网格调参。候选权重优先覆盖 dG、lig_dens、shape_overlap、center_distance_proxy、ligand_size_penalty、virtual node penalty。每组权重都要把 cost matrix 送入 Hungarian，然后用 evaluation 指标比较。

第三步，实现 tree ensemble/ranking 路线。优先检查服务器是否已有 LightGBM、XGBoost、CatBoost；如果没有，使用 sklearn RandomForestClassifier、HistGradientBoostingClassifier 或 GradientBoostingRegressor。若支持 ranking objective，按 group 训练；否则先用 pair-level label 训练，再按 group 排序验证。

第四步，实现 assignment-level validation。把模型 score 转成 cost，重新跑 Hungarian，输出样本级 match accuracy、RMSD <= 2/3/5 Å、真实 assignment cost、当前最优 cost、rank/top k/top k%。

第五步，输出比较报告。报告必须同时展示简单网格和强模型的结果，按样本、ligand 类型、receptor 来源、前置 center hit 档位分层。

## Concrete Steps

本地验证：

    cd C:\Users\15919\OneDrive\My_Project\Pocket_Plus
    python -m compileall Docking

服务器运行预期：

    cd /home/penghongen/My_Project/Pocket_Plus
    source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
    python Docking/ml_scoring/train_pair_ranker.py --eval-run-id 20260520_eval_full80_partial --model-run-id 20260520_ranker_v1 --jobs 48

## Validation and Acceptance

验收标准：

- 训练脚本记录使用的 feature 列、排除的 oracle 列、样本数、group 数和随机种子。
- 网格 baseline 至少输出每组权重的 assignment-level 指标。
- tree ensemble 路线至少输出 feature importance 或等价解释信息。
- 任一模型结果都必须回填 Hungarian 后验证，不能只报告 pair-level AUC 或 accuracy。
- 报告必须展示强模型是否超过手工 cost 和简单网格 baseline。

## Idempotence and Recovery

模型输出必须写入 `/home/penghongen/分子对接尝试/model_runs/{model_run_id}`。重复运行同一 `model_run_id` 只能覆盖该目录下自己生成的文件。训练失败不能破坏 evaluation tables。若某个 ML 包不可用，记录到 `package_check.json` 并使用下一个可用实现。

## Artifacts and Notes

需要上网或查官方文档时，优先确认各库当前 Python API 的 ranking objective 和安装状态。具体模型选择以服务器环境可用包为准，不为了第一版强行修改全局环境。

## Interfaces and Dependencies

预期脚本：

- `Docking/ml_scoring/build_feature_table.py`
- `Docking/ml_scoring/grid_search_cost.py`
- `Docking/ml_scoring/train_pair_ranker.py`
- `Docking/ml_scoring/evaluate_model_assignment.py`

这些脚本的输入应是 evaluation run 输出的 CSV/JSON；输出应是 model run 目录中的 predictions、metrics 和 model metadata。

## Revision Notes

2026-05-20: 初始 ML scoring 子计划落盘。根据用户反馈，路线改为少变量网格与 tree ensemble/ranking 并行，并把 assignment-level objective 纳入主验证。
