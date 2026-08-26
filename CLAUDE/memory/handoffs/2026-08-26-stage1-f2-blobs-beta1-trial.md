# Handoff: Stage1 F2 blobs 的 beta=1 basic 增量实验完成

Date: 2026-08-26

## Current State

主工作树已完成 `unet_c1` 的“F2 语义 blobs + beta=1 basic 调参”增量实验。
实验复用原 macro 结果根中的 F2 blobs，只新增参数文件
`F2_basic_beta1.json` 和 calibration/validation 的
`f2_blobs_basic_beta1_selected` 评估。作业已正常结束，全部本地改动保持
unstaged；开始任务前已有的 `训练与运行/sh/unet_c1.sh` 修改未被触碰。

## Completed

- 新增 `ops/stage1_f2_beta1_trial` 直线编排入口、README、计划和执行记录。
- 使用 Slurm 临时目录承接现有 CLI 的默认 `F2_basic.json` 文件名，再把选择
  结果原子发布为 `F2_basic_beta1.json`；正式的 beta=2 文件始终未被覆盖。
- Job `356956` 使用 16 CPU、64 GiB 完成 calibration 调参与两个数据划分的
  evaluate，终态为 `COMPLETED/0:0`。
- 新参数为 `score_threshold=0.5382110476493835`、`min_voxels=24`、
  `prefiltered_min_voxel=8`，calibration macro F1 三项目标为
  1.3114210328347076。
- calibration/validation 分别核对 100/200 个逐 PDB NPZ 与相同行数 JSONL，
  两份汇总均含 micro、macro 与 PRAUC。
- 本地和远端正式推理 CPU 回归分别通过 54 项测试。

## Decisions

- 实验不修改正式 Stage1 Python 接口，不重算 probability、F2 语义阈值或 F2
  blobs，也不生成 centered。
- 新结果与原 F1/F2 经典结果同目录并存，以显式文件名区分，不增加哈希或身份
  检查。
- 联合分析保存在隔离 Li 工作树，主工作树只保存本实验自己的执行事实，避免
  复制 Li 日志。
- 未获用户明确许可前，不提交、暂存或整理本轮 Git 历史。

## Open Questions

- 是否把经典 F2-beta1 作为后续默认 basic 方案；它在 validation 上与经典 F1
  的 macro F1 三项和近乎相同，并取得最高 macro F2 三项和。
- 用户是否批准提交本增量实验的 ops、文档与 handoff。

## Next Actions

1. 等待用户审阅结果与工作树改动。
2. 只有获得明确 commit 许可后，才处理本轮 unstaged 文件。
3. 如需改变正式默认方案，先单独确认文件名、调用入口和需重算的 producer 范围。

## Files To Reopen

- `ops/stage1_f2_beta1_trial/README.md`
- `ops/stage1_f2_beta1_trial/PLAN.md`
- `ops/stage1_f2_beta1_trial/EXECUTION.md`
- `ops/stage1_f2_beta1_trial/run_trial.sh`
- `C:\Users\15919\Desktop\Pocket_Plus_stage1_li_ratio_trial\ops\stage1_li_ratio_trial\ANALYSIS.md`
