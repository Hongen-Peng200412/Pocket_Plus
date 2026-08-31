# Handoff: Find_1 生产实现已验收，H100 Job 366071 待真实动力学门禁

Date: 2026-09-01

## Current State

Find_1 的生产实现位于工作树 `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra`、分支 `codex/find1-training`。PDB-centric-1 与 PDB-centric-2 的独立 Dataset、训练、实验和 shell 配置，以及同一个 AdamW 内 voxel/trunk 与 point 两组分别按范数 0.5 裁剪的实现已经完成。表达/结构与科学逻辑各一轮全面审查均已完成，针对报告项的窄口径复核均已通过；完整可收集测试为 `367 passed, 11 warnings in 75.05s`。当前只剩生产提交、隔离部署、真实 GPU 优化动力学门禁与正式训练。PDB-centric-2 只允许配置与测试，本轮不得提交训练。

H100 Job `366071` 已于 2026-09-01 01:41:37 +08:00 在 `hnode02` 获得 1 张 H100 和 32 CPU。旧 attempt a1 使用尚未更新的共享代码，在模型实例化阶段被本任务具名 `kill_lock_366071` 终止，退出码为 137，没有完成优化器 step，也不是有效的 PDB-centric-1 训练结果。allocation runner 已消费 `kill_lock` 并于 01:43:14 创建根级 `try_lock_366071`。截至 01:51:57，Job 仍为 RUNNING，`try_lock` 与 `after_lock` 均存在，因此 H100 资源被安全保留但没有继续执行代码。

## Resource Identity

- Job：`366071`，Slurm 状态为 RUNNING，节点为 `hnode02`，资源为 H100×1、CPU×32。
- 根级暂停锁：`/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_366071`，mtime 为 2026-09-01 01:43:14 +08:00。
- allocation 保留锁：`/home/penghongen/Feedback/Pocket_Plus/allocations/366071/after_lock_366071`，mtime 为 2026-09-01 01:41:37 +08:00。
- `pre_lock_366071` 与 `kill_lock_366071` 均不存在。
- allocation 日志：`/home/penghongen/Feedback/Pocket_Plus/allocations/366071/out` 与 `/home/penghongen/Feedback/Pocket_Plus/allocations/366071/err`。
- attempt a1 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_5b8ca0631e1c/Pocket_Plus`。
- attempt a1 launch：`/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T014214_a1`。
- attempt a1 run：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-CPC1/Find_1-CPC1____Find_1_job366071_20260901T014214_a1_CPC1`。
- 动态命令：`/home/penghongen/Feedback/Pocket_Plus/allocations/366071/run_cmd_366071.sh`，a1 时 SHA-256 为 `feef2e59312130978c1dc298c51915a6c19259b067fead9f29e57327259db7b3`。
- a1 `config.yaml` SHA-256：`3a0d342ece0e09214d084c8a0256cd6fe1c68a8102bb168ff655dfb43404da0a`。
- a1 `train.yaml` SHA-256：`4ca1c9b05f6ba313df92ba43736089d0303fce532f02813a108c5d1aade422d7`。

attempt a1 的准确命令、监视器载荷、锁时间和路径均记录在 `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md` 的“2026-09-01 01:41—01:51 H100 Job 366071 获得资源并回到 try_lock”一节。监视器等待脚本曾把 `try_lock` 错误地查到 allocation 子目录，因此本地 helper 最终超时；该错误不影响 `kill_lock` 已被 runner 消费、根级 `try_lock` 已创建这一服务器事实。

## Implementation Facts

- 生产分支基点为当时唯一最新的 `Learn/CUMULATIVE` 提交 `275fcec06ad...`；跨节点 DDP 基础设施提交为 `1d231f27...`。本轮生产改动已经通过审查与最终测试，准备提交真实实现历史。
- `src/wrappers/voxel_point_stage1.py` 通过 `gradient_clip_mode=find_voxel_point` 启用两组裁剪。voxel/trunk 集合按已核实的 AUTO B_trunk 可训练参数前缀定义，point 集合是其余全部可训练参数；仍然只有一个 AdamW。
- PDB-centric-1 固定为每卡 batch 6、全局 batch 48、24 workers、70 epochs、每个 epoch 12 次 validation、学习率 `5e-5`、warmup 比例 `0.005`。
- PDB-centric-2 固定为每卡 batch 6、全局 batch 48、24 workers、110 epochs、每个 epoch 8 次 validation；它只完成配置与测试。
- 最终本地验证已通过：完整可收集集合 `367 passed, 11 warnings in 75.05s`；Python 编译、两份 Find_1 shell 语法和 `git diff --check` 均为退出码 0。`tests/test_stage1_producers.py` 因基点缺少既有 `src.artifacts` 模块而在收集阶段失败，与本轮实现无关。
- AUTO 当前保留最优仍是 B_trunk，score 为 `1.6670933207345788`。参考 checkpoint 为 `/storage/penghongen/tmp/AUTO/TUNE--Find-v3-macro/trials/baseline-7aae1f126ee0-20260828T034121771-b6/training/logs/Pocket_Plus_AUTO_tune_stage1_v3_macro/baseline-7aae1f126ee0-20260828T034121771-b6____baseline-7aae1f126ee0-20260828T034121771-b6/checkpoints/BEST.ckpt`，SHA-256 为 `8b00e3def507119a8e39473d64533617c340922146a392ed1defc0e36547e770`。
- 动力学对照临时工具位于 `tmp/find1_optimization_dynamics/`。该工具只进入真实实现提交和服务器门禁，最终学习端点不得保留临时工具。

## Historical Resume

历史续训已经在隔离工作树 `C:\Users\15919\Desktop\Pocket_Plus_worktrees\find1_historical_resume`、分支 `codex/find1-historical-resume` 完成并保持干净。基点为 `1315d301c867c99e2dc0736feffde86e9cd7fa0a`；依次提交跨节点 DDP、完整 checkpoint 恢复与首轮 sampler 跳过、稳定 kill_lock 信号测试，以及跨目录 `ModelCheckpoint` 状态迁移，端点提交为 `5b4aa5f52d699d904fa82e97e22099047e607e4c`。目标是字面上的 `last.ckpt`：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt`

其 SHA-256 为 `d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c`，记录 epoch 0、global step 11287 和 45150 个已完成的 rank-local microbatches。`45150 = 11287 × 4 + 2`，按用户决策允许丢弃尚未完成一次 optimizer step 的两个累计梯度。

## Monitoring Contracts

- A800 监视代理必须每 300 秒检查一次 `nvlink` 的全部 PENDING 作业。只要 `nvlink` 没有任何 PENDING 作业，就现场推算取消 Job `350302` 和 `356946` 后两个节点可用 CPU，先提交并核实带 `pre_lock` 与 `after_lock` 的双节点 A800 新 Job，最后才取消且只取消这两个旧 Job。接管成功后由主任务继续部署和放行。
- H100 监视代理继续核查 Job `366071` 和节点级 GRES。只有同一 H100 节点确有两张可分配 H100 且 H100 没有等待作业，才可以先取得双卡任务再考虑停止单卡。当前节点级 `GRES_USED` 为 3/3，且有其他用户作业 `366156` 等待，因此不具备双卡切换条件。
- 主任务进入排队、正常 step 或长时间加载等稳定状态后，不使用 heartbeat；每个 60 或 90 分钟静默观察周期由连续的 `Start-Sleep -Seconds 300` 组成，以便最多 5 分钟响应用户新消息。
- 不记录无状态变化的轮询；作业启动、attempt 变化、release/launch/run 变化、checkpoint、验证、错误与修复等明确事件必须写入执行记录并更新 handoff。

## Next Actions

1. 提交生产实现真实历史，把已验收版本部署到服务器隔离目录并核对哈希；不得改写共享 `/home/penghongen/My_Project/Pocket_Plus`。
2. 在 Job `366071` 的 `try_lock` 仍存在时，改写其具名动态命令，使下一 attempt 依次运行八条 AUTO/完整 Find_1 轨迹命令与四条比较命令并返回 `try_lock`。比较命令必须使用从两个只读 release 独立核验的 `src/` 与 `configs/` 内容摘要。门禁通过后，才改为正式 PDB-centric-1 训练命令并删除 `try_lock_366071`。
3. A800 监视代理获得资源后，部署历史续训端点 `5b4aa5f52d699d904fa82e97e22099047e607e4c`、核对 checkpoint 与代码身份，再删除新任务的 `pre_lock`；不得操作未授权作业。
4. 训练正常推进后持续监视，并在首个 optimizer step、首次 validation、checkpoint 或故障等关键事件更新执行记录和 handoff。

## Files To Reopen

- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `src/wrappers/voxel_point_stage1.py`
- `tests/test_find1_gradient_clipping.py`
- `tmp/find1_optimization_dynamics/README.md`
- `训练与运行/sh/Find_1.sh`
- `训练与运行/sh/Find_1_pdb_centric_2.sh`
