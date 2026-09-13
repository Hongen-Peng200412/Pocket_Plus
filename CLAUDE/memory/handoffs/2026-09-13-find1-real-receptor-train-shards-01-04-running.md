# Handoff: Find_1 真实受体训练集第 1—4 片已启动

Date: 2026-09-13

## Current State

Job `368455` 在 hnode01 提供 H100×2 与 64 CPU。第 8 次执行已于 2026-09-13 17:31:45 启动固定训练集 50 片中的用户第 1、2、3、4 片：GPU 0 顺序处理第 1、2 片，GPU 1 顺序处理第 3、4 片。当前第一轮的两个 probability 进程分别处理 CLI 下标 0 和 2，即用户第 1、3 片；两张 H100 利用率为 97% 与 100%，未见本轮新增错误。

根目录 `try_lock_368455` 已按用户授权删除，Job 子目录中的 `after_lock_368455` 仍存在。不得删除该 `after_lock`，不得使用 `scancel`。只有正式入口成功或失败后，runner 才会重新创建 `try_lock_368455`。

2026-09-14 00:59 本地 Codex 服务发生重启。01:00 恢复守护后的远端核验确认任务未中断：Job、release、launch 和 probability Python PID `317866/317867` 均连续不变；两个进程已运行约 7 小时 29 分，第 1、3 片合计已有 235 份 probability 完成标记；两卡利用率为 99% 与 100%。allocation 的 `out`/`err` 字节数及修改时间仍与正式启动瞬间一致，`try_lock_368455` 不存在，`after_lock_368455` 存在。

## Completed

- 四片身份门控已通过：训练清单共有 13,717 个唯一 PDB，前四片各 275 个，共 1,100 个；片内、片间均无重复，完整 50 片无重复且覆盖清单全集。
- 训练清单 SHA-256 为 `8e7f975ea49ee94e6819f2b35bf596c9bc4abacac9afc0aaebe2018b90d94d00`。门控 manifest 为 `/storage/penghongen/tmp/find1_real_receptor_train_shards_20260913/preflight.json`，SHA-256 为 `0efd579fda05029125ee087cae993aa8cb09d966b58fd5bc22efc69e22227c74`。
- 实现端点为 `0e5787298a2e97bb56cc80633786da4aee5bb2b0`；学习端点与 `Learn/CUMULATIVE` 为 `2b15853edecdd8b4193ee38a1c5e77e49f4f9945`，其标题为 `——————开始训练集真实受体推理——————`。两端任务文件逐字节一致，学习端点只额外包含用户原样暂存的 `talk/global/global_9.12.md`。
- Windows 与 Linux allocation 内的三个相关测试文件均为 50 项通过。代码与 Git 布局、中文注释和科学逻辑三类审查均完成两轮全面核查及问题对应的窄复核。
- 旧动态命令 SHA-256 为 `bf42a829532d82ece3e22d6b8ca5ef6960debf56032621f02fb3a4617c64a13a`；新动态命令 SHA-256 为 `ebfa6b4483a4ced4ba8f8580e6291f748d00e43a8b4e5b2701354de50b5a4dbd`。两者分别保存在服务器临时证据目录的 `run_cmd_368455.preimage.sh` 和 `run_cmd_368455.launch.sh`。
- release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_8160a29cc7ca/Pocket_Plus`，内容 SHA-256 为 `8160a29cc7cad4b2405ce269a8bfd899bea5ef522150fa8cc1d304349bd5554d`。
- launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260913T173145_a8`。launch 保存的 H100×2、64 CPU、hnode01、release 和动态命令均已核验。

## Decisions

- checkpoint 固定为 `TOP_epoch_04_score_0.6654.ckpt`，SHA-256 为 `3f5dd715da76a2337ddb94017f442783798566cac584b8dd16882d8afd792ee4`；内部 `global_step=38623` 对应 W&B 已完成步编号 38,622。
- 复用 calibration 冻结的 F2 语义阈值 `0.431304931640625` 和 `objective_beta=1` Gaussian 参数。不得在训练集重新调参。
- 每片固定执行 probability → F2 blobs → centered → Gaussian score-only；`forward_min_voxels=8`，超大 blob 继续前向，候选顺序、来源索引、四组 offsets 和数组字段不变。
- 正式产物根为 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train`。已有完整产物按完成标记复用，score-only 对同一 centered 文件幂等重算 `score/selected`。

## Next Actions

1. 稳定运行期间按用户要求，以连续 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待；不得创建 heartbeat。
2. 醒来后核验 Job、锁、两个 GPU 进程和各阶段完成数量。只在失败、从第 1/3 片切换到第 2/4 片、锁状态变化或全部完成时更新执行记录与 handoff。
3. 正式入口结束后，按 manifest 的 1,100 个成员验收 probability、F2 blobs、F2 centered、`score/selected`、候选顺序与 offsets；检查四片无遗漏、无重复且没有额外训练 PDB 目录。
4. 即使任务完成也必须保留 `after_lock_368455`，让 allocation 停回 `try_lock_368455`。

## Files To Reopen

- `文档/exec_plan/2026-09-13_Find_1真实受体训练分片推理.md`
- `训练与运行/sh/infer/find1_real_receptor_train_shards.sh`
- `tmp/find1_real_receptor_train_shards_20260913/capture_shards.py`
- `/storage/penghongen/tmp/find1_real_receptor_train_shards_20260913/preflight.json`
- `CLAUDE/memory/handoffs/2026-09-12-find1-real-receptor-evaluation-running.md`
