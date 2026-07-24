# Handoff: AdaLigand Stage1 Find_2 实现与单 H200 交接

Date: 2026-07-21

## Current State

Find_1 Gaussian 改版与 Find_2 56D tune 已完成本地实现、配置、统一 producer 下游适配和双 subagent 独立审计。服务器尚未同步这些新改动，也尚未提交两个单 H200 Job；旧 Job 321108 仍以双 H200 健康运行，Find_0 Job 321106 与 unet_c1 Job 321107 也继续运行。

活执行日志是 `talk/Excx_执行stage1端到端训练.md`，Find_2 冻结设计依据是相邻 AdaLigand 仓库的 `grill_with_memory/07-21-12-07.md`。

## Completed

- `src/model/stage1_embed_head.py` 新增默认关闭的 `use_gaussian_splatting` 和 `voxel_embed_as_tune`。Find_1 保留 49D raw residual 与 2D occupancy，Find_2 固定 56D、无 voxel raw residual/occupancy。
- `src/model/stage1_model.py` 让 Find_2 的 density56 与 Gaussian voxel tune56 逐元素相加；Find_0/Find_1/unet_c1 旧路径保持原语义。
- 新增 Find_2 embed、CPC1、CPC2、Selector 配置；Find_2 CPC2 保持从同名 CPC1 BEST 严格 model-only 初始化。
- `src/stage1_producers.py` 成为四 producer 唯一名单，并由其派生 Find 子集；Dataset、artifact、inference、CLI 与 Selector 已统一复用。
- 本地配置/Gaussian 测试 15 passed；下游 subagent 验证 51 passed；独立审计下游 73 passed，配置/producer 21 passed。审计发现并修复 `tests/test_cpc_v3_configs.py` 的 Find_2 分类遗漏。
- 本机完整模型测试仅因 `baseline_env` 缺少 `torch_cluster` 无法收集，必须在服务器正式环境补跑。
- 新增 run-scoped 单 H200 allocation、双 Job 提交与 smoke→formal 包装，且 `tmp/stage1_formal_find_chain.sh` 已支持 Find_2。

## Decisions

- 新版 Find_1 与 Find_2 各使用单 H200、micro-batch 6、global batch 48、gradient accumulation 8。
- 两个新 sbatch 同时提交；只要至少一个真正 `RUNNING`，立即精确 `scancel 321108`。若两分钟内两个均未运行，则取消两个新 Job 并保留 321108。
- 新链先在 allocation 快照内运行服务器定向测试和真实 5 optimizer-step smoke；只有真实 OOM 或代码错误才恢复，不因正式训练期间显存百分比停止。
- Find_0 Job 321106 与 unet_c1 Job 321107 不得因本次替换而停止。

## Open Questions

- 服务器正式环境需复跑核心/下游定向测试，并复核本地成熟 Selector `np.linalg.eigvalsh` 原生 abort 是否只限 Windows BLAS/LAPACK 环境。
- 新版 Find_1/Find_2 的 5-step smoke、W&B 初始化、50-step 稳定门槛与 30 分钟密集监控尚未完成。

## Next Actions

1. 非删除式安全同步 Pocket_Plus 到 `/home/penghongen/My_Project/Pocket_Plus`。
2. 在服务器执行新脚本 `bash -n`，随后运行 `tmp/adaligand_stage1_submit_find12_single_h200.sh`。
3. 高频检查两个新 Job；至少一个 `RUNNING` 后精确取消旧 321108，并把新 Job ID 写入 ExecPlan。
4. 用 `tmp/adaligand_stage1_launch_single_find_smoke_and_formal.sh JOB_ID Find_1|Find_2` 解除各自 `pre_lock`，监控服务器定向测试、5-step smoke 和正式 CPC1。
5. 两条新 run 均达到至少 50 optimizer step且密集监控至少 30 分钟后，更新或重建每 3 小时 heartbeat `adaligand-stage1`。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `src/model/stage1_embed_head.py`
- `src/model/stage1_model.py`
- `src/stage1_producers.py`
- `tmp/adaligand_stage1_submit_find12_single_h200.sh`
- `tmp/adaligand_stage1_launch_single_find_smoke_and_formal.sh`
- `tmp/stage1_formal_find_chain.sh`
