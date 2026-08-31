# Stage1 v3 数据准备执行记录

本文记录 `talk/global.md` 中“基础建设——数据”和“基础设施——划分”的一次性实现与服务器运行证据。本记录覆盖完整体数组迁移、新 split、第三版 BOX 几何候选池，以及 2026-08-24 从既有候选池冻结的 PDB 中心 validation selection；Dataset、模型与训练入口的消费适配另见 AdaLigand 的对应执行记录。

## 实现基点与隔离边界

- 实现共同基点：`Learn/CUMULATIVE@f2216171f8279dcee4bb09cba168540e340d67f4`。
- 实现分支：`codex/stage1-v3-data-preparation`。
- 实现端点：`bbaebe008d42e89923375eeaad98d055b43e030f`。
- 学习分支：`Learn/stage1-v3-data-preparation`，端点为 `da659b999eaa70aac26a9ead221a129e4434dd04`。
- 用户验收后的累计端点：`Learn/CUMULATIVE@186bfd05b90f6cbe3fe165e16068ac4bcdc39c7f`。该提交只补充用户对代码复杂度和可读性的批注，没有重新生产或改变服务器数据。
- 独立工作树：`C:\Users\15919\.codex\worktrees\stage1-v3-data-preparation\Pocket_Plus`。
- 目录边界：新增实现、测试、运行入口和文档全部位于 `ops/stage1_data_preparation/`；没有修改 `src/`、配置、模型或现有 `ops/box_pool_2/`。

## 当前进度

- 四类完整体数组已经完成同目录、可重试原子迁移，并通过全量复核。
- EMDB 发布时间续传日志、严格日期/质量/资产划分和 200 validation/100 calibration/剩余 train 已经冻结。
- 第三版 0:5:5 BOX pool 已从迁移后的 `exp.npz:canonical_shape_zyx` 读取形状完成构建和验收。
- Windows 本地测试已运行，初次发现 Windows 无法用 POSIX 方式同步目录及测试夹具缺少 `ligand_area.npz:schema_version=3`；修正后 5 项测试通过。
- 独立严格审查已给出 `APPROVED`：批准 12 个 Slurm 数组任务、每任务 9 核的正式迁移，并逐项批准了原子迁移、外部访问、资产门禁、科学产物构造和命令行边界所需的函数拆分例外。

## 服务器执行证据

- 同步范围：只同步 `/home/penghongen/My_Project/Pocket_Plus/ops/stage1_data_preparation/`，未使用删除参数；服务器端 5 项测试通过，所有 shell 入口通过 `bash -n`。
- 完整体数组迁移：Slurm 数组 Job `343572`，数组范围 `0-11%12`，每个数组元素 9 CPU，总并发 108 CPU；12 个数组元素全部 `COMPLETED 0:0`。最终复核 Job `343835` 为 `COMPLETED 0:0`。
- 迁移结果：检查 22,381 个密度目录和 89,524 个目标位置；89,442 个来源字段已迁移，82 个目标位置因来源文件不存在而记为 `source_absent`。正式 NPY 总字节数为 13,813,200,929,408。运行摘要位于 `/storage/penghongen/AdaLigand/Ori_Data/reports/runs/stage1_npy_migration_20260817_v1/summary.json`。
- 划分冻结：Job `345237` 为 `COMPLETED 0:0`。来源包含 22,251 个 PDB、662,078 条 Stage G 候选；审计得到 14,017 个可训练 PDB、2,497 个日期留出 PDB、357 个缺日期隔离 PDB、3,718 个质量淘汰 PDB、1,655 个资产无效 PDB、4 个缺文件 PDB和 3 个短图 PDB。
- 正式划分：train 为 13,717 个 PDB、451,505 条候选；validation 为 200 个 PDB、6,104 条候选；calibration 为 100 个 PDB、2,426 条候选；held-out 为 2,497 个 PDB、81,922 条候选；缺日期隔离集合为 357 个 PDB、11,327 条候选。
- 第三版 BOX pool：108 CPU 数组 Job `346035` 的 12 个数组元素全部 `COMPLETED 0:0`；最终复核 Job `346063` 为 `COMPLETED 0:0`。train 发布 13,717 个 PDB，validation 发布 200 个 PDB，两个集合均无零 context PDB；验证选择固定包含 16,525 个 bias 与 16,525 个 context 条目，center 条目为 0。
- 旧版 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2` 未被改写；本轮产物全部位于 `stage1_preparation_box_pool_3`。

## 计划与实现差异

- 有益差异：第三版 BOX pool 不调用仍要求 `exp.npz:grid` 的旧批量入口，而只复用稳定的几何函数，从迁移后保留的 `canonical_shape_zyx` 获取完整图形状。
- 中性差异：EMDB 发布时间先写入可续传 JSONL，再冻结 split；这不改变日期口径。
- 有害差异：尚未发现。
- 未完成范围：新版推理程序不在本轮数据准备范围内。Dataset、模型和训练入口后来已由独立训练 I/O 任务适配，但正式训练仍须等待用户明确授权。

## 2026-08-24 PDB 中心验证选择

- Pocket_Plus 实现分支 `codex/stage1-pdb-centric-sampling` 从 `Learn/CUMULATIVE@8561d2790614a0d92f0ad88ebce241a9f1c77ba3` 建立；首个稳定实现提交为 `ac53172`。
- 最终安全同步后，服务器 `src/datasets/stage1_requests.py` 与冻结脚本的文件 SHA-256 分别为 `de9621f1640567c9f3fc8fb31532619dd8e75f38e78b08dcdd75c39c248718b2` 与 `9a20a4744442f5f7e6ec6da0cb1db7218788de134f2c933a0779755259e950bd`，与 Windows 工作树字节一致。Git 以 LF 行尾保存的对应 blob 内容 SHA-256 分别为 `a31413473e1c539bc8398d84fd2531faaab232bb0e861bed6cf77ed76242e14c` 与 `4987560d4e19ba5df53b99e69dd81b49d79f4b4096adfb0900c6dd3203bdc318`；两组差异仅来自 CRLF/LF 行尾。
- 正式命令为 `python -m ops.stage1_data_preparation.freeze_validation_selection_pdb_centric`。最终文件于 2026-08-25 00:02:04 +08:00 覆盖写入，保存从原 200 个 validation PDB 中按 `SeedSequence(3407, spawn_key=(2,))` 无放回选出的 150 个身份，以及 3,750 个 bias、3,750 个 context 和 0 个 center 请求。
- 新文件精确包含八类既有索引数组与三个采样参数标量，共 11 个字段；逐字段 dtype、shape、候选索引范围和逐 PDB 计数均通过核验。150 个 PDB 都恰好包含 25 个 bias 与 25 个 context，PDB 身份按原 validation manifest 顺序保存。
- 第三轮逻辑审查发现原 `SeedSequence([3407, 2])` 与第 3 个 manifest PDB 的 occurrence 排列随机状态碰撞。改用独立 `spawn_key` 后，最终集合相对碰撞版本保留 116 个身份并替换 34 个身份；该中间版本未用于训练。
- 新文件大小为 71,150 字节，SHA-256 为 `546ebd3a1f07b230af42911b6740f466c6af289c8bff91a387c4eb8b1d69dd8e`；重复执行正式命令后哈希不变。
- 原 `validation_selection.npz`、`manifest.json`、`config.json`、`summary.json` 与 `_COMPLETE` 的修改时间分别保持在 2026-08-18 或 2026-08-17；本次没有改写这些 V3 产物。对应 SHA-256 分别为 `91af9c01538e6da1a0a11d2109eeb28673b97cf4c89b1e7e813f7f2b843f5ac7`、`fcfa65c0ac58116eb1f65009baa0eb2e03df94f1306629ae9232a286b1041e2e`、`4fccabf4d75d40afc6e6dd8a13f7e5dd89d8d11576f77a5b31c7ee13723342de`、`2b49dfed736583ba8f7a81138e66ff0c9af7209a02216a783dd3acca487dc031` 与空文件哈希 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- 最终定向回归 50 项通过。除基线遗留的 `tests/test_stage1_producers.py` 外，Windows 全量回归为 345 项通过、11 条 warning；该遗留文件在共同基点 `8561d2790614a0d92f0ad88ebce241a9f1c77ba3` 上同样因不存在的 `src.artifacts` 包而在收集阶段失败，本轮没有扩大范围修复。
- Python 编译检查、`submit_task.sh` 与五个训练 Shell 的 `bash -n`、两个仓库的 `git diff --check` 均通过。对 occurrence 数量 `O=1..1000` 的独立数学审计确认每个 PDB 分配总数为 25、任意两个 occurrence 的分配数之差不超过 1、单个 occurrence 不超过 cap 25，完整轮转周期内累计分配相等。
- 代码布局与 Git、中文注释、科学逻辑三类独立审查各完成三轮全面核查；第三轮整改后的三类窄口径复核均为 `APPROVED`。最终冻结脚本使用的验证 PDB 随机域、资源说明、配置测试、函数布局、Docstring 和字段注释均已纳入复核。
- 本次没有提交、取消、重启或修改任何 GPU Job。

## 2026-08-26 采样方式三 V2 验证选择

- 新增的硬编码生产入口为 `python -m ops.stage1_data_preparation.freeze_validation_selection_pdb_centric_2`。入口读取 validation manifest 的全部 200 个 PDB，以 `50/0.7575757575757576/1` 参数冻结 epoch 0，只写 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric_v2.npz`。
- V2 产物保存 200 个 PDB、3,305 个 bias、3,200 个 context 和 0 个 center 请求，共 6,505 个 BOX。11 个字段的名称、顺序、dtype 与形状均通过一次性服务器核验；七组索引均落在对应 PDB、occurrence 和候选范围内，三个采样标量分别为 `50`、`0.7575757575757576` 和 `1`。生产请求展开结果同样为 3,305 个 bias 与 3,200 个 context。
- V2 文件的实际 SHA-256 为 `0c92a731a7676f8083a250ff54e13dd2356e2cc462e383f6ad45c4c30de5fda2`。该哈希只作为本次执行证据记录，没有写入 Python、YAML、shell 或测试代码。
- 运行冻结入口前后，V1 `validation_selection_pdb_centric.npz` 的 SHA-256 均为 `546ebd3a1f07b230af42911b6740f466c6af289c8bff91a387c4eb8b1d69dd8e`，证明 V2 生产没有读取后改写、删除或覆盖 V1。
- Windows 本地定向回归 52 项通过，数据准备目录回归 10 项通过；Hydra 组合得到 V2 路径与 `50/0.7575757575757576/1`，Python AST、`unet_c1.sh` 与 `submit_task.sh` 的 `bash -n`、`git diff --check` 均通过。本轮保持进入任务前的暂存区索引不变，没有暂存或提交实现。

## 2026-08-28 采样方式三 unet_c1 正式训练

- 正式提交 Job 为 `358384`。提交资源为单节点、1 张 H100 和 32 CPU，`pre_hold=0`、`after_hold=1`；Job 于 2026-08-28 09:31:26 +08:00 在 `hnode01` 启动，`after_lock_358384` 保持存在。
- 启动留证为 `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260828T093205_a1/launch.json`，release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_58b27dd765ed/Pocket_Plus`。
- 正式运行目录为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal`；W&B run 为 `pencounkdual-111/AdaLigand_Stage1/9dsi7i9w`。
- 最终 `config.yaml` 已核对：`devices=1`、`nnodes=1`、`num_workers=30`、V2 验证文件、训练采样参数 `50/0.7575757575757576/1`、`max_epochs=110`、`val_per_epoch=8`、`warmup_ratio=0.005`、每卡 batch 8、全局 batch 48、学习率 `1e-4`，蛋白与核酸主链损失权重均为 `0.05`。
- 从该 release 的生产请求入口重新展开得到每个 epoch 436,211 个训练 BOX，其中 bias 为 216,739 个、context 为 219,472 个；V2 验证请求为 6,505 个 BOX，其中 bias 为 3,305 个、context 为 3,200 个。
- release 与运行目录冻结快照中的 `src/datasets/stage1_requests.py` SHA-256 均为 `74ffcbbf916d0758b64da084312932db93b1e93b7b2f69648955b62df4bfe514`，证明正在训练的请求逻辑来自本次 release。
- 稳定性验收期间，W&B 摘要从 `trainer/global_step=200` 推进到 `278`；训练总损失与受体、配体体素、蛋白主链、核酸主链和配体距离损失均为有限数值。H100 抽样利用率为 89%，显存使用量为 79,848/81,559 MiB；未发现 traceback、CUDA OOM、DataLoader 失败或非有限值错误。
- 2026-08-28 14:12 +08:00，首次完整 validation 已结束。验证总损失为 `0.3082846`，配体体素 PRAUC 为 `0.2585518`，受体 PRAUC 为 `0.2138680`；受体、配体体素、蛋白主链、核酸主链和配体距离验证损失均为有限数值。
- 首次 validation 生成 `checkpoints/TOP_epoch_00_score_0.2586.ckpt` 与 `checkpoints/last.ckpt`，文件大小均约 499.7 MB。任务随后继续推进到 `trainer/global_step=1304`，没有发现 traceback、CUDA OOM、DataLoader 失败或非有限值错误。
- 2026-08-28 18:50 +08:00，第二次完整 validation 已结束。验证总损失降至 `0.2699645`，配体体素 PRAUC 升至 `0.3804999`，受体 PRAUC 升至 `0.3469226`；新的 `TOP_epoch_00_score_0.3805.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=2288`。
- 2026-08-28 23:30 +08:00，第三次完整 validation 已结束。验证总损失为 `0.2663394`，配体体素 PRAUC 为 `0.4109082`，受体 PRAUC 为 `0.3935394`；新的 `TOP_epoch_00_score_0.4109.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=3686`。
- 第三次 validation 后，稳定别名 `BEST.ckpt` 已从首次 TOP 刷新为第二次的 `TOP_epoch_00_score_0.3805.ckpt`，说明训练中的别名比最新 TOP 晚一个 validation 回调。最新高分模型始终明确保存在独立 TOP 文件中；最终退出验收必须确认 `BEST.ckpt` 已追上最终最佳模型，当前不需要停止训练或改写 checkpoint。
- 2026-08-29 04:05 +08:00，第四次完整 validation 已结束。验证总损失降至 `0.2514138`，配体体素 PRAUC 升至 `0.4482627`，受体 PRAUC 升至 `0.4602929`；新的 `TOP_epoch_00_score_0.4483.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=4673`。`BEST.ckpt` 同步刷新为上一轮的 `TOP_epoch_00_score_0.4109.ckpt`，别名时序与前述结论一致。
- 2026-08-29 08:39 +08:00，第五次完整 validation 已结束。验证总损失降至 `0.2458982`，配体体素 PRAUC 升至 `0.4758326`，受体 PRAUC 升至 `0.4872300`；新的 `TOP_epoch_00_score_0.4758.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=5957`。`BEST.ckpt` 已刷新为第四次的 `TOP_epoch_00_score_0.4483.ckpt`。
- 2026-08-29 13:15 +08:00，第六次完整 validation 已结束。验证总损失降至 `0.2383705`，配体体素 PRAUC 为 `0.4785158`，受体 PRAUC 升至 `0.5158617`；新的 `TOP_epoch_00_score_0.4785.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=7307`。`BEST.ckpt` 已刷新为第五次的 `TOP_epoch_00_score_0.4758.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-29 17:50 +08:00，第七次完整 validation 已结束。验证总损失降至 `0.2311407`，配体体素 PRAUC 为 `0.4713585`，受体 PRAUC 升至 `0.5202941`；新的 `TOP_epoch_00_score_0.4714.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=8012`。`BEST.ckpt` 已刷新为第六次的 `TOP_epoch_00_score_0.4785.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-29 22:23 +08:00，第八次完整 validation 已结束。验证总损失为 `0.2300247`，配体体素 PRAUC 创新高至 `0.5097033`，受体 PRAUC 创新高至 `0.5372400`；新的 `TOP_epoch_00_score_0.5097.ckpt` 与更新后的 `last.ckpt` 已生成，任务进入 epoch 1 并继续推进到 `trainer/global_step=9149`。`BEST.ckpt` 暂时保持第六次的 `TOP_epoch_00_score_0.4785.ckpt`，符合已确认的回调时序；未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-30 03:08 +08:00，第九次完整 validation 已结束。验证总损失降至 `0.2178248`，配体体素 PRAUC 创新高至 `0.5327364`，受体 PRAUC 创新高至 `0.5641335`；新的 `TOP_epoch_01_score_0.5327.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=10289`。`BEST.ckpt` 已刷新为第八次的 `TOP_epoch_00_score_0.5097.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-30 07:57 +08:00，第十次完整 validation 已结束。验证总损失为 `0.2191286`，配体体素 PRAUC 为 `0.5293481`，受体 PRAUC 创新高至 `0.5670031`；新的 `TOP_epoch_01_score_0.5293.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=11366`。`BEST.ckpt` 已刷新为第九次的 `TOP_epoch_01_score_0.5327.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-30 12:45 +08:00，第十一次完整 validation 已结束。验证总损失为 `0.2243590`，配体体素 PRAUC 小幅创新高至 `0.5349470`，受体 PRAUC 创新高至 `0.5844466`；新的 `TOP_epoch_01_score_0.5349.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=13079`。`BEST.ckpt` 暂时保持第九次的 `TOP_epoch_01_score_0.5327.ckpt`，符合已确认的回调时序；未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-30 17:31 +08:00，第十二次完整 validation 已结束。验证总损失为 `0.2182586`，配体体素 PRAUC 创新高至 `0.5367544`，受体 PRAUC 为 `0.5783800`；新的 `TOP_epoch_01_score_0.5368.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=13742`。`BEST.ckpt` 已刷新为第十一次的 `TOP_epoch_01_score_0.5349.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-30 22:14 +08:00，第十三次完整 validation 已结束。验证总损失为 `0.2181289`，配体体素 PRAUC 为 `0.5267326`，受体 PRAUC 为 `0.5686539`；新的 `TOP_epoch_01_score_0.5267.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=14846`。`BEST.ckpt` 已刷新为第十二次的 `TOP_epoch_01_score_0.5368.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 2026-08-31 02:57 +08:00，第十四次完整 validation 已结束。验证总损失为 `0.2088470`，配体体素 PRAUC 为 `0.5241200`，受体 PRAUC 为 `0.5801458`；新的 `TOP_epoch_01_score_0.5241.ckpt` 与更新后的 `last.ckpt` 已生成，validation 对应 `trainer/global_step=15902`。
- 2026-08-31 07:42 +08:00，第十五次完整 validation 已结束。验证总损失为 `0.2180344`，配体体素 PRAUC 创新高至 `0.5419699`，受体 PRAUC 创新高至 `0.5948478`；新的 `TOP_epoch_01_score_0.5420.ckpt` 与更新后的 `last.ckpt` 已生成，validation 对应 `trainer/global_step=17038`。
- 2026-08-31 12:29 +08:00，第十六次完整 validation 已结束。验证总损失降至 `0.2037673`，配体体素 PRAUC 创新高至 `0.5507095`，受体 PRAUC 创新高至 `0.6137356`；新的 `TOP_epoch_01_score_0.5507.ckpt` 与更新后的 `last.ckpt` 已生成，任务进入 epoch 2 并继续推进到 `trainer/global_step=18176`。`BEST.ckpt` 已刷新为第十五次的 `TOP_epoch_01_score_0.5420.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 第十四至第十六次 validation 期间，本地睡眠会话因宿主挂起延长；服务器 Job 始终保持 `RUNNING`。恢复后通过 Slurm、W&B `scan_history` 和 checkpoint 时间戳补齐了三次验证证据，训练本身未受影响。
- 2026-08-31 17:12 +08:00，第十七次完整 validation 已结束。验证总损失为 `0.2076540`，配体体素 PRAUC 为 `0.5448655`，受体 PRAUC 创新高至 `0.6180575`；新的 `TOP_epoch_02_score_0.5449.ckpt` 与更新后的 `last.ckpt` 已生成，任务继续推进到 `trainer/global_step=19574`。`BEST.ckpt` 已刷新为第十六次的 `TOP_epoch_01_score_0.5507.ckpt`，未发现 traceback、CUDA OOM、NCCL、DataLoader 失败或非有限值错误。
- 后续采用由多个 300 秒命令组成的 60 或 90 分钟静默守护周期，只在新的 validation、checkpoint、故障恢复、训练退出或其他明确状态变化时继续更新本记录。

## 接续位置

后续训练或推理适配不得重复构造 split 或 BOX pool，也不得重新写入迁移后的 NPZ。采样方式三的活动科学说明位于 `talk/global/global_8.26.md`；Job `358384` 的当前状态、恢复边界和后续检查项记录在 `CLAUDE/memory/handoffs/2026-08-28-stage1-pdb-centric-v2-unet-c1-running.md`。
