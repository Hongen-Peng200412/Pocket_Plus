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

attempt 4 已完成 100/100 个 calibration probability，并冻结 F1 语义阈值 `0.733612060546875`、F1 basic 参数 `score_threshold=0.7737352252006531`、`prefiltered_min_voxel=8`、`min_voxels=13`，以及 F2 语义阈值 `0.431304931640625`；F1/F2 blobs 均为 100/100。首次 F2 centered 前向在两张卡上都因输入缺少 `atom_label` 抛出 `KeyError`，centered NPZ 与完成标记均为 0。attempt 以退出码 1 回到 `try_lock_368455`，`after_lock_368455` 始终保留。

根因边界已经核实：无监督 centered 请求按 Dataset 契约不读取真实 binding 标签；当前 Find 训练快照只用 `atom_label` 的 dtype 和真实原子轴构造 P anchor 占位，前向特征和本任务评分不消费标签值。最小修复是在 centered 推理批次内补充与 `atom_global_indices` 对齐的全假 bool 占位；不得改成读取 calibration 或测试标签。

该修复已经进入正式 attempt 5 并通过真实前向验证：

- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d24c9e99c664/Pocket_Plus`；
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260912T230920_a5`；
- 启动时间：2026-09-12 23:09；只删除了 `try_lock_368455`，`after_lock_368455` 保留；
- 23:17 已完成 34/100 个 calibration `F2_centered.npz` 及对应 `_COMPLETE`，两个 centered 分片均存活，显存约为 44.6/43.2 GiB，没有新 traceback 或 OOM。
- 23:37 已完成 100/100 个 calibration F2 centered，并冻结 Gaussian 参数：`score_threshold=0.9002149105072021`、`lambda_positive=0.064`、`lambda_negative=0.0128`、`tau_angstrom=1.25`、`min_voxels=27`、`prefiltered_min_voxel=8`、`objective_beta=1`。正式入口已进入 179-PDB `test_0` probability，首批 2 个 PDB 的四件套产物完整；双 H100 利用率为 100%/99%，无失败标记或新异常。
- 2026-09-13 06:07，attempt 5 正式成功并停回 `try_lock_368455`。`test_0` 的 probability、F1/F2 blobs、F2 centered 和两套逐 PDB evaluation NPZ 均为 179/179；基础与 Gaussian 全局 JSONL/metrics 已生成，`after_lock_368455` 保留。
- 06:24 的计算节点只读门控已通过：PDB 成员与顺序、完成标记、文件集合、冻结选择掩码、`forward_min_voxels=8` 的 centered 保序子序列、offsets、macro 均值、micro F1、top-K 计数/分母及正式聚合重算全部一致。正式产物拓扑 SHA-256 为 `62d8d6bd43a7c80e3a9c7295fa559e28be39e847d405d7ddd03f5d462ddfa5f9`。
- 06:27 原子切换为一次性 `derive_test1.py` 正式命令并只删除 try lock，启动 attempt 6；release 仍为 `Pocket_Plus_d24c9e99c664`，launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260913T062754_a6`。
- 06:29 attempt 6 成功并停回 `try_lock_368455`。两套 `test_1` 各有 149 行 JSONL、metrics 和 provenance；成员顺序、逐行父记录等价、聚合关系、全部 provenance 哈希及父产物哈希均通过最终门控。`held_out_test_1` 没有 probability、blobs、centered 或逐 PDB evaluation NPZ。

## Frozen Scientific Identity

- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume/checkpoints/TOP_epoch_04_score_0.6654.ckpt`；
- checkpoint SHA-256：`3f5dd715da76a2337ddb94017f442783798566cac584b8dd16882d8afd792ee4`；
- checkpoint 内部 `global_step=38623`，对应 W&B 已完成步编号 `38622`；
- resolved config SHA-256：`9a4ea5cf95d50deb3d42c8420e4dddf1d3a7eb0ebb6d0ee4edbff0388f1884da`；
- calibration/test_0/test_1 清单数分别为 100/179/149，SHA-256 分别为 `b14c9f44...0fe7a`、`12473392...c887`、`ee0697aa...0df0`。

基础路径为 F1 semantic blobs、basic 调参、`objective_beta=1` 和完整 `test_0` 评估。Gaussian 路径复用 probability，使用 F2 semantic blobs、centered 前向、Gaussian 调参、`objective_beta=1` 和完整 `test_0` 评估。任何 PDB 即使超过 1,000 个 blob 也只记录标识，不被排除；centered 只前向来源体素数至少为 8 的 blob。

## Code And Release Identity

Learn/CUMULATIVE 已快进到 Find centered 修复的学习端点 `b76c61ede6f8373e972875eaea203d53c9761f41`。对应实现端点 `1aeeb2217362617f183201914b4a6d2fe139155d` 与学习端点的 Git tree 均为 `78ae22176fea95a3fd063034e82aae0d097c190c`。修复只在缺键时补充与 `atom_global_indices` 同形的全假 bool `atom_label`；47 项定向测试及 Git/代码布局、中文注释、科学逻辑三类窄复核均已批准。

release 中的关键 SHA-256：

- `src/inference/centered.py`：`ec13ac7b46d5db42e9523d6ba9781a24c5b4d1a231516c187cf312d18497856b`；
- `src/inference/evaluation.py`：`da076afb3cb75efda1c1f615c472cc5dbf55bd37b72fc7ce7e98440595249999`；
- `configs/inference/stage1_v3.yaml`：`937aecc3b473c2caf415685583f1a6793c7bb228dd650c0844010a1178395f80`；
- `find1_real_receptor_evaluation.sh`：`95e4afe5925a7a8cf389ccb803b5be36e00fa1ed7e3bb37b052c522185a5e754`；
- `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`：`72e0e2d6d4715e1f9d4a702cf281a3496a86227f16e536fe4f27d2d8bceb3984`。

## Next Actions

1. 第一阶段已经完成。核对 `talk/global/global_9.12.md` 第 (3) 项与现有官方 score-only 入口，冻结 validation 清单数量及 SHA-256，并建立最短正式编排命令。
2. calibration 只做冻结 Gaussian score-only；validation 依次做 probability、冻结阈值 F2 blobs、centered 和冻结 Gaussian score-only，不生成 cal/val 评估。完成实现门控和必要审查后，原子改写动态命令并只删除 try lock。
3. 始终保留 `after_lock_368455`，未经用户新授权不触碰该锁或执行 `scancel`。仅在启动稳定、失败、try_lock、阶段完成或最终完成等关键事件更新执行记录与本 handoff。

## Files To Reopen

- `文档/exec_plan/2026-09-12_Find_1真实受体推理与评估.md`
- `训练与运行/sh/infer/find1_real_receptor_evaluation.sh`
- `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`
- `CLAUDE/memory/handoffs/2026-09-12-find1-pdb-centric-training-stage-complete.md`
