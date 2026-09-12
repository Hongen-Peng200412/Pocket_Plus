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

用户预先授权的 batch 24 到 18 回退已经完成。attempt 4 于 2026-09-12 19:10 触发：

- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_bcb995435809/Pocket_Plus`；
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260912T191102_a4`；
- 完整图 batch 为 18，centered batch 仍为 12，每个 GPU 进程仍使用 26 个请求物化线程；
- release 中配置 SHA-256 为 `937aecc3b473c2caf415685583f1a6793c7bb228dd650c0844010a1178395f80`，其余四个关键文件哈希与 attempt 3 相同。

截至 20:26，batch 18 已通过首批真实产物验收：calibration 中 41/100 个 PDB 分别具有一份 `probability_map.npz`、`geometry.json`、`performance.json` 和 `_COMPLETE`，没有失败标记。两张 H100 显存约 74.08 GiB、利用率 100%，两个分片进程均存活；`try_lock_368455` 不存在，`after_lock_368455` 始终保留。

## Frozen Scientific Identity

- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume/checkpoints/TOP_epoch_04_score_0.6654.ckpt`；
- checkpoint SHA-256：`3f5dd715da76a2337ddb94017f442783798566cac584b8dd16882d8afd792ee4`；
- checkpoint 内部 `global_step=38623`，对应 W&B 已完成步编号 `38622`；
- resolved config SHA-256：`9a4ea5cf95d50deb3d42c8420e4dddf1d3a7eb0ebb6d0ee4edbff0388f1884da`；
- calibration/test_0/test_1 清单数分别为 100/179/149，SHA-256 分别为 `b14c9f44...0fe7a`、`12473392...c887`、`ee0697aa...0df0`。

基础路径为 F1 semantic blobs、basic 调参、`objective_beta=1` 和完整 `test_0` 评估。Gaussian 路径复用 probability，使用 F2 semantic blobs、centered 前向、Gaussian 调参、`objective_beta=1` 和完整 `test_0` 评估。任何 PDB 即使超过 1,000 个 blob 也只记录标识，不被排除；centered 只前向来源体素数至少为 8 的 blob。

## Code And Release Identity

Learn/CUMULATIVE 已快进到 OOM 回退学习端点 `3a546325059e8d120aa1d8ffff939a6624a0ffd3`。对应实现端点 `96ae78cc021aaaac1aafc280e44e3a4e21373518` 与学习端点的 Git tree 均为 `7be35a4a2a92e0f1f5373862e3395b46691fc352`。原实现的两轮三角色全面审查和遗留问题窄复核全部批准；batch 18 窄修复再次通过 47 项定向测试、shell 语法、YAML 资源断言和补丁格式门控，未重新扩大审查范围。

release 中的关键 SHA-256：

- `src/inference/evaluation.py`：`2c966c9933abfde81ec6c08236a5a86a497e901b191a7aade07c5150655c72cd`；
- `configs/inference/stage1_v3.yaml`：`937aecc3b473c2caf415685583f1a6793c7bb228dd650c0844010a1178395f80`；
- `find1_real_receptor_evaluation.sh`：`95e4afe5925a7a8cf389ccb803b5be36e00fa1ed7e3bb37b052c522185a5e754`；
- `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`：`72e0e2d6d4715e1f9d4a702cf281a3496a86227f16e536fe4f27d2d8bceb3984`。

## Next Actions

1. 当前继续执行一次由连续 `Start-Sleep -Seconds 300` 组成的 60 分钟静默等待；醒来后核对 calibration 是否完成以及流程是否进入 F1/F2 blobs、调参或 held-out probability。
2. attempt 4 成功完成两套 `test_0` 评估并重新创建 `try_lock_368455` 后，先核对 179-PDB probability、blobs、centered、逐 PDB评估及两份汇总，再原子改写动态命令为执行记录中的一次性 `derive_test1.py` 命令，仅删除该 try lock 触发下一次 attempt。
3. 派生 attempt 结束后核对两套 149-PDB JSONL、metrics 与 provenance。始终保留 `after_lock_368455`，未经用户新授权不触碰该锁或执行 `scancel`。
4. 仅在启动稳定、OOM/失败、try_lock 或全部结果完成等关键事件更新执行记录与本 handoff。

## Files To Reopen

- `文档/exec_plan/2026-09-12_Find_1真实受体推理与评估.md`
- `训练与运行/sh/infer/find1_real_receptor_evaluation.sh`
- `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`
- `CLAUDE/memory/handoffs/2026-09-12-find1-pdb-centric-training-stage-complete.md`
