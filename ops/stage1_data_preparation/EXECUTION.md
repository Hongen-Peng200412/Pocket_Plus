# Stage1 v3 数据准备执行记录

本文记录 `talk/global.md` 中“基础建设——数据”和“基础设施——划分”的本轮实现与服务器运行证据。当前只覆盖完整体数组迁移、新 split 和第三版 BOX pool；Dataset、模型、训练和推理均不在本轮修改范围内。

## 实现基点与隔离边界

- 实现共同基点：`Learn/CUMULATIVE@f2216171f8279dcee4bb09cba168540e340d67f4`。
- 实现分支：`codex/stage1-v3-data-preparation`。
- 独立工作树：`C:\Users\15919\.codex\worktrees\stage1-v3-data-preparation\Pocket_Plus`。
- 目录边界：新增实现、测试、运行入口和文档全部位于 `ops/stage1_data_preparation/`；没有修改 `src/`、配置、模型或现有 `ops/box_pool_2/`。

## 当前进度

- 已建立四类数组的同目录、可重试原子迁移实现与 12×9 CPU Slurm 数组入口。
- 已建立 EMDB 发布时间续传日志、严格日期/质量/资产划分和 200 validation/100 calibration/剩余 train 的冻结实现。
- 已建立从迁移后 `exp.npz:canonical_shape_zyx` 读取形状的第三版 0:5:5 BOX pool 实现与 12×9 CPU Slurm 数组入口。
- Windows 本地测试已运行，初次发现 Windows 无法用 POSIX 方式同步目录及测试夹具缺少 `ligand_area.npz:schema_version=3`；修正后 5 项测试通过。
- 独立严格审查已给出 `APPROVED`：批准 12 个 Slurm 数组任务、每任务 9 核的正式迁移，并逐项批准了原子迁移、外部访问、资产门禁、科学产物构造和命令行边界所需的函数拆分例外。

## 服务器执行证据

尚未填写。正式提交、Job 编号、分片状态、split 计数和 BOX pool 验收会在实际发生后记录于此。

## 计划与实现差异

- 有益差异：第三版 BOX pool 不调用仍要求 `exp.npz:grid` 的旧批量入口，而只复用稳定的几何函数，从迁移后保留的 `canonical_shape_zyx` 获取完整图形状。
- 中性差异：EMDB 发布时间先写入可续传 JSONL，再冻结 split；这不改变日期口径。
- 有害差异：尚未发现。
- 未完成范围：服务器环境验证、108 核全量迁移、正式 split、108 核 BOX pool 和最终双线 Git 收口。
