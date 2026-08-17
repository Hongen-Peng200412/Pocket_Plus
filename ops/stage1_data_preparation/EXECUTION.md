# Stage1 v3 数据准备执行记录

本文记录 `talk/global.md` 中“基础建设——数据”和“基础设施——划分”的一次性实现与服务器运行证据。本记录自身只覆盖完整体数组迁移、新 split 和第三版 BOX pool；后续 Dataset、模型与训练入口的消费适配另见 AdaLigand 的 `文档/exec_plan/Stage1第三版训练IO与入口实施.md`。

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

## 接续位置

后续训练 I/O 适配不得重复构造 split 或 BOX pool，也不得重新写入迁移后的 NPZ。当前第三版产物和下一轮实施停点记录在 AdaLigand 的 `文档/exec_plan/Stage1第三版训练IO与入口实施.md`；完整接手信息记录在 `CLAUDE/memory/handoffs/2026-08-17-stage1-v3数据准备完成与训练IO接续.md`。
