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
- 最终代码安全同步后，服务器 `src/datasets/stage1_requests.py` 与冻结脚本的 SHA-256 分别为 `013ae573dc0520b3779e340020787919ee809e3289dca05c8f45fbabe01684ff` 与 `4987560d4e19ba5df53b99e69dd81b49d79f4b4096adfb0900c6dd3203bdc318`，与本地一致。
- 正式命令为 `python -m ops.stage1_data_preparation.freeze_validation_selection_pdb_centric`。最终文件于 2026-08-25 00:02:04 +08:00 覆盖写入，保存从原 200 个 validation PDB 中按 `SeedSequence(3407, spawn_key=(2,))` 无放回选出的 150 个身份，以及 3,750 个 bias、3,750 个 context 和 0 个 center 请求。
- 新文件精确包含八类既有索引数组与三个采样参数标量，共 11 个字段；逐字段 dtype、shape、候选索引范围和逐 PDB 计数均通过核验。150 个 PDB 都恰好包含 25 个 bias 与 25 个 context，PDB 身份按原 validation manifest 顺序保存。
- 第三轮逻辑审查发现原 `SeedSequence([3407, 2])` 与第 3 个 manifest PDB 的 occurrence 排列随机状态碰撞。改用独立 `spawn_key` 后，最终集合相对碰撞版本保留 116 个身份并替换 34 个身份；该中间版本未用于训练。
- 新文件大小为 71,150 字节，SHA-256 为 `546ebd3a1f07b230af42911b6740f466c6af289c8bff91a387c4eb8b1d69dd8e`；重复执行正式命令后哈希不变。
- 原 `validation_selection.npz`、`manifest.json`、`config.json`、`summary.json` 与 `_COMPLETE` 的修改时间分别保持在 2026-08-18 或 2026-08-17；本次没有改写这些 V3 产物。对应 SHA-256 分别为 `91af9c01538e6da1a0a11d2109eeb28673b97cf4c89b1e7e813f7f2b843f5ac7`、`fcfa65c0ac58116eb1f65009baa0eb2e03df94f1306629ae9232a286b1041e2e`、`4fccabf4d75d40afc6e6dd8a13f7e5dd89d8d11576f77a5b31c7ee13723342de`、`2b49dfed736583ba8f7a81138e66ff0c9af7209a02216a783dd3acca487dc031` 与空文件哈希 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- 最终定向回归 50 项通过。除基线遗留的 `tests/test_stage1_producers.py` 外，Windows 全量回归为 345 项通过、11 条 warning；该遗留文件在共同基点 `8561d2790614a0d92f0ad88ebce241a9f1c77ba3` 上同样因不存在的 `src.artifacts` 包而在收集阶段失败，本轮没有扩大范围修复。
- Python 编译检查、`submit_task.sh` 与五个训练 Shell 的 `bash -n`、两个仓库的 `git diff --check` 均通过。对 occurrence 数量 `O=1..1000` 的独立数学审计确认每个 PDB 分配总数为 25、任意两个 occurrence 的分配数之差不超过 1、单个 occurrence 不超过 cap 25，完整轮转周期内累计分配相等。
- 代码布局与 Git、中文注释、科学逻辑三类独立审查各完成三轮全面核查；第三轮整改后的三类窄口径复核均为 `APPROVED`。最终冻结脚本使用的验证 PDB 随机域、资源说明、配置测试、函数布局、Docstring 和字段注释均已纳入复核。
- 本次没有提交、取消、重启或修改任何 GPU Job。

## 接续位置

后续训练 I/O 适配不得重复构造 split 或 BOX pool，也不得重新写入迁移后的 NPZ。当前第三版产物和下一轮实施停点记录在 AdaLigand 的 `文档/exec_plan/Stage1第三版训练IO与入口实施.md`；完整接手信息记录在 `CLAUDE/memory/handoffs/2026-08-17-stage1-v3数据准备完成与训练IO接续.md`。
