# Stage1 v3 数据准备工具

本目录提供三类可复用运维工具：把四个完整体数组从压缩 NPZ 迁移到同目录 NPY；按 EMDB 首次发布时间、质量和资产契约冻结 Stage1 v3 划分；从迁移后元数据生成 0:5:5 BOX pool。工具不修改 `src/`、模型、Dataset 或第二版 BOX pool。

## 目录与稳定入口

```text
ops/stage1_data_preparation/
├── atomic_io.py
├── migrate_density_arrays.py
├── freeze_split.py
├── build_box_pool_3.py
├── utils/
│   ├── __init__.py
│   └── box_pool.py
├── run/
│   ├── migrate_density_array.sh
│   ├── finalize_density_migration.sh
│   ├── fetch_and_freeze_split.sh
│   ├── build_box_pool_3.sh
│   ├── finalize_box_pool_3.sh
│   └── sync_ops.ps1
├── tests/
├── README.md
└── EXECUTION.md
```

正式服务器数据根目录为 `/storage/penghongen/AdaLigand/Ori_Data`。所有重型步骤通过项目根目录的 `训练与运行/submit_task.sh` 进入 Slurm；不要在 SSH 登录会话中直接运行迁移或 BOX pool 构造。

`run/sync_ops.ps1` 只把本目录上传到服务器 Pocket Plus 项目的同名 `ops/` 目录；它不删除远端文件、不传输主代码，并强制使用已经固定的 SSH 主机密钥。

## 完整体数组迁移

每个已有来源 NPZ 独立迁移；某个 PDB 缺少其中一种来源文件时记录 `source_absent`，不创建虚构产物。

| 原字段 | 同目录目标文件 | 数组契约 | 迁移后的原 NPZ |
| --- | --- | --- | --- |
| `exp.npz:grid` | `exp.npy` | `float32 (1,D,H,W)`，实验密度，空间轴为 ZYX | 保留 `grid` 外全部字段 |
| `sim.npz:grid` | `sim.npy` | `float32 (1,D,H,W)`，模拟密度，空间轴为 ZYX | 保留 `grid` 外全部字段 |
| `ligand_dist.npz:distance` | `ligand_dist.npy` | `float16 (1,D,H,W)`，最近配体原子距离 | 保留 `distance` 外全部字段 |
| `ligand_area.npz:union_mask` | `union_mask.npy` | `bool (1,D,H,W)`，全部 occurrence 配体区域并集 | 保留 `union_mask` 外全部字段 |

单文件事务顺序如下：

1. 从原 NPZ 解出目标数组，并复制所有剩余字段。
2. 在同目录写临时 NPY，执行文件同步，再以内存映射逐块核对形状、数据类型和全部数值。
3. 原子发布 NPY并同步目录。
4. 在同目录写压缩临时 NPZ，重新打开并逐字段核对，再原子替换原 NPZ并同步目录。

进程若在两次替换之间退出，目录会同时保留有效 NPY 和仍含原字段的 NPZ；重试会验证 NPY 后完成 NPZ 替换。NPZ 已无目标字段但 NPY 缺失、已有 NPY 与来源字段不一致等状态均为硬错误，不会自动覆盖。

迁移运行记录位于 `/storage/penghongen/AdaLigand/Ori_Data/reports/runs/stage1_npy_migration_20260817_v1`：

- `shards/shard_XXX_of_012.json`：每个数组元素负责的 PDB 和四类文件状态。
- `summary.json`：PDB 数、目标文件状态计数和 NPY 总字节数。
- `_COMPLETE`：全部分片集合与正式目录复查成功后最后发布的空文件。

## Stage1 v3 划分

`freeze_split.py fetch-release-dates` 从 EMDB 官方 `entry/admin/{emdb_id}` 接口读取 `admin.key_dates.map_release`。`emdb_release_dates.jsonl` 是可续传日志；每个 JSON object 包含 `emdb_id`、`map_release`、`status` 和 `source_url`。

`freeze_split.py freeze` 使用以下严格规则：

- 一个 PDB 在 `raw/pair_list.jsonl` 中可能对应多个 EMDB；最早的非空 `map_release` 是首次发布时间。
- 首次发布时间 `< 2026-01-01` 才进入非 held-out 候选；等于或晚于该日期的 PDB 全部进入 `held_out.json`，不按质量过滤。
- 缺少首次发布时间的 PDB 进入 `quarantine_missing_release.json`。
- 非 held-out 候选记录必须同时满足 `map_resolution < 4.0` 与 `cc_contour > 0.65`；等于阈值不通过。
- PDB 必须具备四个迁移后 NPY、四个元数据 NPZ、`receptor_tokens.npz` 和 `atom_labels.npz`，且四个体数组的形状、数据类型和空间几何一致。
- 完整图 ZYX 三轴均不小于 80 才能进入 train、validation 或 calibration。
- 合格 PDB 按 `sha256(3407|eval|pdb_id)` 升序排列；前 200 个属于 validation，随后 100 个属于 calibration，其余全部属于 train。

划分目录 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split` 包含：

- `train.json`、`validation.json`、`calibration.json`：只保存逐条通过质量规则的 Stage G 候选记录，同一 PDB 不跨文件。
- `held_out.json`：日期达到界线的 PDB 的全部 Stage G 候选记录。
- `quarantine_missing_release.json`：无法确定首次发布时间的 PDB 的全部 Stage G 候选记录。
- `pdb_audit.jsonl`：每个候选 PDB 的首次发布时间、最终状态、完整图形状、失败细节和通过质量条件的候选数量。
- `config.json`：阈值、日期边界、稳定排名规则和输入位置。
- `summary.json`：来源计数、审计状态计数和各集合的 PDB/候选记录数。
- `_COMPLETE`：上述文件全部原子发布后的完成标记。

## Stage1 v3 BOX pool

BOX pool 根目录为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool`。每个 `train/<pdb_id>.npz` 或 `validation/<pdb_id>.npz` 包含：

- `pdb_id`：字符串标量，当前 PDB identity。
- `occurrence_id`：`int32 (N_occ,)`，升序 occurrence 编号。
- `center_start_zyx`：`int32 (N_occ,3)`，每个 occurrence 的居中 BOX 起点；请求比例为 0，因此只保留兼容字段。
- `bias_start_zyx`：`int32 (N_occ,30,3)`，经验体积半径与额外 0–3 Å 独立扰动生成的候选起点。
- `context_start_zyx`：`int32 (N_context,3)`，完整图逐轴合法范围内均匀采样的共享起点，最多 500 个，不设置核心受体原子数量门槛。

根目录还包含 `manifest.json`、`validation_selection.npz`、`config.json`、`summary.json` 和最后发布的 `_COMPLETE`。训练请求比例固定为 center:bias:context=`0:5:5`；每个 PDB 每轮最多选择 50 个 occurrence 的运行规则不由本目录修改。

## 2026-08-17 正式运行结果

- 数组迁移：22,381 个密度目录，89,524 个目标位置；89,442 个来源字段成功迁移，82 个目标位置没有来源文件。新发布 NPY 共 13,813,200,929,408 字节。
- 冻结划分：train 13,717 PDB，validation 200 PDB，calibration 100 PDB，日期留出 2,497 PDB，缺日期隔离 357 PDB。
- BOX pool：train 和 validation 分别发布 13,717 与 200 个 PDB NPZ；两者均无零 context PDB。固定验证选择包含 16,525 个 bias 与 16,525 个 context，center 为 0。
- Slurm 证据：迁移数组/复核为 Job `343572`/`343835`，划分为 Job `345237`，BOX pool 数组/复核为 Job `346035`/`346063`；全部以退出码 `0:0` 完成。
- 第二版 `stage1_preparation_box_pool_2` 没有被读取或改写。详细计数、路径和审查结论见同目录 `EXECUTION.md`。

## 108 核运行顺序

迁移和 BOX pool 分别使用 12 个 Slurm 数组元素，每个元素申请 9 核，最大同时占用 108 核：

```bash
bash 训练与运行/submit_task.sh --simple \
  --sh ops/stage1_data_preparation/run/migrate_density_array.sh \
  --resource cpu --cpus 9 --array 0-11%12
```

迁移数组全部成功后运行 `finalize_density_migration.sh`；随后运行 `fetch_and_freeze_split.sh`；再以相同 12×9 配置运行 `build_box_pool_3.sh`，最后运行 `finalize_box_pool_3.sh`。每一步都以前一步的完成标记和文件级验收为前提。

## 验证

Windows 本地测试使用按环境名解析出的 `Pocket_Plus_windows` Conda 环境：

```powershell
python -m pytest -q ops/stage1_data_preparation/tests
```

测试覆盖迁移逐值一致、剩余字段不变、幂等重试、冲突 NPY 拒绝覆盖、缺失 NPY 硬失败、日期与质量严格边界、200/100/剩余划分无交叉，以及 `exp.npz` 已无 `grid` 时 BOX pool 仍可确定性构造。

## 训练消费状态

第三版产物现在由唯一的 `src/datasets/stage1_dataset.py::Stage1Dataset` 直接消费。四个完整体 NPY 以只读 mmap 延迟打开，训练请求固定为 `0:5:5`；数据准备函数不进入 Dataset 的生产依赖方向。`utils/box_pool.py` 只供本目录的 V3 构建入口复用，集中 occurrence mask、确定性 PDB seed、bias/context 起点和 validation selection 冻结逻辑。
