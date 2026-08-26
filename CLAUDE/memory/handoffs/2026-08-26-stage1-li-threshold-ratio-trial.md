# Handoff: Stage1 Li 阈值与候选比例实验完成

Date: 2026-08-26

## Current State

隔离工作树 `C:\Users\15919\Desktop\Pocket_Plus_stage1_li_ratio_trial` 已完成
`unet_c1` 的逐 PDB Li 阈值实验。calibration 100 个 PDB 用于拟合 F1/F2
basic_ratio 参数，validation 200 个 PDB 只冻结复用这些参数。服务器正式产物
全部位于名称以 `--Li` 结尾的独立结果根，代码和文档仍为 unstaged，没有任何
commit。

## Completed

- 实现逐 PDB Li 阈值、1/32768 向上量化、26 连通区域和精确候选比例搜索。
- F1 选出比例 0.3391224863、`min_voxels=37`；F2 选出比例
  0.7596153846、`min_voxels=37`。
- calibration Job `356954` 与 validation Job `356955` 均正常完成。
- calibration/validation 分别发布 100/200 个 Li blobs 与两套完整评估；两套
  PRAUC、micro、macro 和 top-K 指标均已核对。
- 已把 Li 两套参数配置与经典 F1、经典 F2、经典 F2-beta1 做联合分析。
- 本地与远端正式推理 CPU 回归均通过，最新结果为 59 项测试通过。

## Decisions

- 当前结果不支持用 Li 阈值替代经典语义阈值。Li-F1 与 Li-F2 在两个数据划分
  的对应 macro 三项目标上均明显落后经典方案。
- Li 仍保留为完整隔离实验事实；不删除产物，也不把实验代码合入正式推理。
- 联合结果中，经典 F2 blobs 加 beta=1 basic 参数在 validation 上最均衡，但
  该观察不自动改变正式科学契约。
- 未获用户明确许可前，不提交、暂存或整理本实验 Git 历史。

## Open Questions

- 用户是否批准保留并提交 Li 实验代码，或仅保留服务器产物后放弃本地实现。
- 是否把经典 F2-beta1 作为后续默认 basic 方案，需要用户单独决定。

## Next Actions

1. 等待用户审阅联合分析与两份工作树状态。
2. 只有获得明确许可后，才处理 isolated worktree 的 commit 或正式接口集成。
3. 若继续新模型实验，沿用同一 calibration 拟合、validation 冻结评估口径。

## Files To Reopen

- `ops/stage1_li_ratio_trial/ANALYSIS.md`
- `ops/stage1_li_ratio_trial/EXECUTION.md`
- `ops/stage1_li_ratio_trial/PLAN.md`
- `ops/stage1_li_ratio_trial/README.md`
- `ops/stage1_li_ratio_trial/run_trial.py`
- `ops/stage1_li_ratio_trial/run_split.sh`
- `tests/inference/test_stage1_li_ratio_trial.py`
