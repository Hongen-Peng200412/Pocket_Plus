# Handoff: Find_1 真实受体训练集第 1—4 片已完成

Date: 2026-09-15

## Current State

Job `368455` 的第 8 次执行已完成固定训练集 50 片中的用户第 1、2、3、4 片：GPU 0 顺序处理第 1、2 片，GPU 1 顺序处理第 3、4 片。四片共 1,100 个唯一 PDB，全部完成 probability、冻结阈值 F2 blobs、centered 和冻结 Gaussian score-only。正式产物根恰好包含清单中的 1,100 个 PDB，没有遗漏、重复或额外目录。

2026-09-15 09:57，正式入口成功返回，runner 重新创建根目录 `try_lock_368455`。Job 子目录中的 `after_lock_368455` 始终存在，因此 Job 仍以 `RUNNING|hnode01|64|gres:gpu:h100:2` 保留双 H100 allocation；正式推理进程已经结束。不得删除该 `after_lock`，不得使用 `scancel`。

2026-09-15 10:36，全量最终验收通过。报告位于 `/storage/penghongen/tmp/find1_real_receptor_train_shards_20260913/final_verification.json`，SHA-256 为 `c6de80244af2db776feffb2739ba732f79c26697ec4d139cea80d6dfc85cfade`。验收逐 PDB 核对三阶段文件、完成标记、数组模式、候选顺序、四组 offsets、超框候选和冻结 Gaussian 精确重算；正式产物根没有 tuning、evaluation、JSONL、metrics、临时文件或失败标记。

## Completed

- 四片身份门控已通过：训练清单共有 13,717 个唯一 PDB，前四片各 275 个，共 1,100 个；片内、片间均无重复，完整 50 片无重复且覆盖清单全集。
- 训练清单 SHA-256 为 `8e7f975ea49ee94e6819f2b35bf596c9bc4abacac9afc0aaebe2018b90d94d00`。门控 manifest 为 `/storage/penghongen/tmp/find1_real_receptor_train_shards_20260913/preflight.json`，SHA-256 为 `0efd579fda05029125ee087cae993aa8cb09d966b58fd5bc22efc69e22227c74`。
- 正式实现的首个端点为 `0e5787298a2e97bb56cc80633786da4aee5bb2b0`；对应的首个学习提交为 `2b15853edecdd8b4193ee38a1c5e77e49f4f9945`，标题严格使用 `——————开始训练集真实受体推理——————`。两端任务文件逐字节一致，学习端点只额外包含用户原样暂存的 `talk/global/global_9.12.md`。
- Windows 与 Linux allocation 内的三个相关测试文件均为 50 项通过。代码与 Git 布局、中文注释和科学逻辑三类审查均完成两轮全面核查及问题对应的窄复核。
- 旧动态命令 SHA-256 为 `bf42a829532d82ece3e22d6b8ca5ef6960debf56032621f02fb3a4617c64a13a`；新动态命令 SHA-256 为 `ebfa6b4483a4ced4ba8f8580e6291f748d00e43a8b4e5b2701354de50b5a4dbd`。两者分别保存在服务器临时证据目录的 `run_cmd_368455.preimage.sh` 和 `run_cmd_368455.launch.sh`。
- release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_8160a29cc7ca/Pocket_Plus`，内容 SHA-256 为 `8160a29cc7cad4b2405ce269a8bfd899bea5ef522150fa8cc1d304349bd5554d`。
- launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260913T173145_a8`。launch 保存的 H100×2、64 CPU、hnode01、release 和动态命令均已核验。
- 最终产物统计为 1,100 份 probability、1,100 份 F2 blobs、1,100 份 F2 centered 和 1,100 份带 `score/selected` 的 centered 文件；共 39,878 个 blobs、38,797 个 centered 候选和 29,452 个冻结阈值入选候选。
- 24 个候选无法由单个 80³ BOX 完整容纳；另有 1 个 PDB 因来源 blobs 总数超过 1,000 而带有 `_BLOB_EXCEED` 标识。验收确认超框候选的完整来源体素数、框内稀疏归档、候选顺序和 Gaussian 分数均符合既定契约，超量 PDB 也未被过滤。

## Decisions

- checkpoint 固定为 `TOP_epoch_04_score_0.6654.ckpt`，SHA-256 为 `3f5dd715da76a2337ddb94017f442783798566cac584b8dd16882d8afd792ee4`；内部 `global_step=38623` 对应 W&B 已完成步编号 38,622。
- 复用 calibration 冻结的 F2 语义阈值 `0.431304931640625` 和 `objective_beta=1` Gaussian 参数。不得在训练集重新调参。
- 每片固定执行 probability → F2 blobs → centered → Gaussian score-only；`forward_min_voxels=8`，超大 blob 继续前向，候选顺序、来源索引、四组 offsets 和数组字段不变。
- 正式产物根为 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train`。已有完整产物按完成标记复用，score-only 对同一 centered 文件幂等重算 `score/selected`。

## Next Actions

1. 本轮第 1—4 片没有剩余运行或验收动作；下批片号只有在用户明确指定后才能启动。
2. 保留 `after_lock_368455` 和当前 `try_lock_368455`。未经用户明确要求，不改写动态命令、不删除锁，也不使用 `scancel`。

## Files To Reopen

- `文档/exec_plan/2026-09-13_Find_1真实受体训练分片推理.md`
- `训练与运行/sh/infer/find1_real_receptor_train_shards.sh`
- `tmp/find1_real_receptor_train_shards_20260913/capture_shards.py`
- `/storage/penghongen/tmp/find1_real_receptor_train_shards_20260913/preflight.json`
- `/storage/penghongen/tmp/find1_real_receptor_train_shards_20260913/final_verification.json`
