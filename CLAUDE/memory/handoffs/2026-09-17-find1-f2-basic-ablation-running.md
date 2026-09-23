# Handoff: Find_1 两种受体条件 F2–F1–basic 消融完成

Date: 2026-09-17

## Current State

真实受体与 CryoAtom2 条件的 F2–F1–basic 消融均已完成。两者各有独立 `F2_basic.json`、179-PDB `test_0` 逐 PDB评估与汇总，以及 149-PDB `test_1` 派生汇总和 provenance；既有 probability、blobs、centered、tuning 与 evaluation 文件均未改写。

## Completed

- 已核对官方 `tune` 在 `score_mode=basic` 时读取 F2 blobs，并以 `source_probability_mean` 打分；官方 `evaluate` 在 `artifact=centered` 时仅在内存重算 basic 分数，不修改 centered NPZ。
- 已完成 558 对 F2 blobs/centered 产物的只读候选轴门控。真实受体 calibration/test_0 分别有 2,792/2,251 个 `voxel_count >= 8` 候选，CryoAtom2 分别有 2,678/2,407 个；blob 编号、完整来源体素数和概率均值逐项一致。
- 已安全同步一次任务代码到服务器共享项目。长期生产 Python、CLI、配置和既有正式结果均未修改。
- 已更新两个实验日志和共享总日志，分别记录正式命令、提交命令、产物边界与当前排队状态。
- 首次提交的 64 CPU Job `384991/384992` 已按用户要求在获得节点前取消；两项运行时间均为零，没有创建 release、launch、预映像或正式产物。替代任务使用一次性 `stage1_v3_cpu8.yaml`，服务器 OmegaConf 门控确认 `calibration.workers=8`，科学搜索网格与评估轴未改变。
- 两项启动前的既有文件快照均包含 3,443 个受保护文件。真实受体 F2-basic 参数为 `score_threshold=0.6524679660797119`、`min_voxels=27`；CryoAtom2 为 `score_threshold=0.7009013295173645`、`min_voxels=26`；共同使用 prefilter 8 和 objective beta 1。
- 23:42 状态快照：真实受体新逐 PDB test_0 评估为 31/179；CryoAtom2 为 179/179，正在全局汇总或 test_1 派生交界。两个 evaluate 进程均持续使用 CPU，错误日志没有 traceback。
- 23:43，CryoAtom2 已写出 `test_0` metrics JSON 与 JSONL，随后在临时 `derive_test1.py` 导入 `scipy` 时失败。窄修复固定一次任务 Python 解释器；恢复入口仅派生 `test_1` 并运行既有文件保护校验，不重复调参或评估。
- CryoAtom2 恢复 Job `385054` 已以 8 CPU 提交；提交前固定解释器的 `scipy` 与正式聚合模块导入门控通过。
- CryoAtom2 恢复 Job `385054` 已成功发布 149-PDB `test_1`，既有 3,443 文件零变化、本次 185 个新增文件完整通过校验。真实受体 Job `385001` 已完成 179-PDB `test_0` 后在同一临时环境位置退出；恢复 Job `385055` 已以 8 CPU 提交。
- 真实受体恢复 Job `385055` 已成功发布 149-PDB `test_1`；两种受体条件均通过 `protected_count=3443`、`new_count=185` 的保护校验。
- 最终只读总门控验证两套 179/149 行结果的顺序、精确子集关系、有限值范围、micro P/R–F1、macro 平均、top-K 和 provenance，状态为 `PASS`。
- AdaLigand 六份本地收口文档已增量加入第三条流水线的完整指标、正式命令、产物路径和派生边界，保持未提交且未上传服务器。

## Decisions

- 调参固定 `alpha=2`、`score_mode=basic`、`objective_beta=1`、`prefiltered_min_voxel=8`，输出 `tuning/F2_basic.json`。
- `test_0` 固定使用 `artifact=centered` 和评估名 `f2_centered_basic_macro_selected`；`test_1` 仅从新 `test_0` 逐 PDB 事实保序派生。
- 不执行任何 GPU 前向，不重新拟合 F2 语义阈值，不运行 basic score-only，不修改既有 Gaussian `score/selected`。
- 一次任务代码仅位于 `tmp/find1_f2_basic_ablation_20260917/`。运行前后以文件大小与纳秒修改时间保护既有 calibration、tuning、test_0 和 test_1 文件。
- 8 CPU 配置只调整 basic 调参的数据读取与事实构造并发数；阈值搜索空间、目标函数、候选分数、测试候选轴和汇总定义均不改变。

## Next Actions

1. 本任务已完成，无待运行的 Slurm Job。
2. 用户审阅 AdaLigand 六份本地收口文档后，再决定是否提交或同步到服务器；当前不要擅自执行这两项操作。

## Files To Reopen

- `tmp/find1_f2_basic_ablation_20260917/run_f2_basic.sh`
- `tmp/find1_f2_basic_ablation_20260917/protect_existing.py`
- `tmp/find1_f2_basic_ablation_20260917/derive_test1.py`
- `tmp/find1_f2_basic_ablation_20260917/stage1_v3_cpu8.yaml`
- `文档/exec_plan/2026-09-12_Find_1真实受体推理与评估.md`
- `文档/exec_plan/2026-09-15_Find_1_CryoAtom2受体推理与评估.md`
- `文档/exec_plan/2026-09-15_Find_1不同受体条件推理与评估总日志.md`
