# Handoff: Find_1 真实受体推理与评估运行中

Date: 2026-09-12

## Current State

Job `368455` 的双 H100、64 CPU allocation 已从 PDB-centric-2 训练接管为 `Find_1` 真实受体推理。训练进程由用户授权的 `kill_lock_368455` 终止；该锁已被 runner 消费并删除。`after_lock_368455` 必须继续保留，不得释放资源或执行 `scancel`。

正式推理 attempt 3 已于 2026-09-12 18:58 启动：

- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_0b517f1ec15f/Pocket_Plus`；
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260912T185836_a3`；
- 正式命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/find1_real_receptor_evaluation.sh"`；
- 正式输出根：`/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1`。

attempt 3 的两个 calibration probability 分片均完成数据加载，但完整图 batch 24 在第一次模型前向时同时发生 CUDA OOM。每张卡当时已有约 71.72 GiB 显存占用，继续申请 11.72 GiB 时只剩 7.16 GiB；attempt 以退出码 1 结束，重新创建了 `try_lock_368455`。正式输出根的文件数仍为 0，因此没有需要清理或混用的半成品；`after_lock_368455` 保持不变。

用户已预先授权 batch 24 OOM 后降到 18。当前应先完成该精确配置回退及 Learn/实现端点等价核验，再安全同步并由新 release 重启，不应直接复用 attempt 3 release。

## Frozen Scientific Identity

- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume/checkpoints/TOP_epoch_04_score_0.6654.ckpt`；
- checkpoint SHA-256：`3f5dd715da76a2337ddb94017f442783798566cac584b8dd16882d8afd792ee4`；
- checkpoint 内部 `global_step=38623`，对应 W&B 已完成步编号 `38622`；
- resolved config SHA-256：`9a4ea5cf95d50deb3d42c8420e4dddf1d3a7eb0ebb6d0ee4edbff0388f1884da`；
- calibration/test_0/test_1 清单数分别为 100/179/149，SHA-256 分别为 `b14c9f44...0fe7a`、`12473392...c887`、`ee0697aa...0df0`。

基础路径为 F1 semantic blobs、basic 调参、`objective_beta=1` 和完整 `test_0` 评估。Gaussian 路径复用 probability，使用 F2 semantic blobs、centered 前向、Gaussian 调参、`objective_beta=1` 和完整 `test_0` 评估。任何 PDB 即使超过 1,000 个 blob 也只记录标识，不被排除；centered 只前向来源体素数至少为 8 的 blob。

## Code And Release Identity

Learn/CUMULATIVE 已快进到 `506f84cf1f013ee58bc70166c50f6c10bda93688`。实现端点 `41c8f4f` 与学习端点的 Git tree 均为 `ae6af1ef56f6092adeb9d30f904b236da0be44d0`。两轮三角色全面审查和遗留问题窄复核全部批准；定向正式测试 47 项通过，临时派生测试 1 项通过。

release 中的关键 SHA-256：

- `src/inference/evaluation.py`：`2c966c9933abfde81ec6c08236a5a86a497e901b191a7aade07c5150655c72cd`；
- `configs/inference/stage1_v3.yaml`：`8b5589db971e94748fc983cb11f3dbe5a47be48f4b9b751205600ba4a7e55b66`；
- `find1_real_receptor_evaluation.sh`：`95e4afe5925a7a8cf389ccb803b5be36e00fa1ed7e3bb37b052c522185a5e754`；
- `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`：`72e0e2d6d4715e1f9d4a702cf281a3496a86227f16e536fe4f27d2d8bceb3984`。

## Next Actions

1. 把 `configs/inference/stage1_v3.yaml` 的完整图 batch 从 24 精确降到 18，同步更新 README 与定向测试；完成双线端点等价和门控后，从 Learn 工作区安全同步。
2. 核对共享目录后保持同一短动态命令，只删除 `try_lock_368455` 启动新 release；确认首个 PDB 正常落盘、无新 OOM 后，再使用多次 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待。
3. 新 attempt 成功完成两套 `test_0` 评估并重新创建 `try_lock_368455` 后，先核对 179-PDB probability、blobs、centered、逐 PDB评估及两份汇总，再原子改写动态命令为执行记录中的一次性 `derive_test1.py` 命令，仅删除该 try lock 触发下一次 attempt。
4. 派生 attempt 结束后核对两套 149-PDB JSONL、metrics 与 provenance。始终保留 `after_lock_368455`，未经用户新授权不触碰该锁或执行 `scancel`。
5. 仅在启动稳定、OOM/失败、try_lock 或全部结果完成等关键事件更新执行记录与本 handoff。

## Files To Reopen

- `文档/exec_plan/2026-09-12_Find_1真实受体推理与评估.md`
- `训练与运行/sh/infer/find1_real_receptor_evaluation.sh`
- `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`
- `CLAUDE/memory/handoffs/2026-09-12-find1-pdb-centric-training-stage-complete.md`
