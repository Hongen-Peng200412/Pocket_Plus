# Stage1 v3 数据准备工具

本目录保存第三版数据准备的完整生产入口：把四个完整体数组从压缩 NPZ 迁移到同目录 NPY；按 EMDB 首次发布时间、质量和资产契约冻结 Stage1 v3 划分；生成逐 PDB bias/context 几何候选池；从既有候选池一次性冻结当前 PDB 中心验证选择。最后一步不重建或改写 V3 候选池，也不修改原 `validation_selection.npz`。

## 目录与稳定入口

```text
ops/stage1_data_preparation/
├── atomic_io.py
├── migrate_density_arrays.py
├── freeze_split.py
├── build_box_pool_3.py
├── freeze_validation_selection_pdb_centric.py
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

根目录还包含 `manifest.json`、`validation_selection.npz`、`config.json`、`summary.json` 和最后发布的 `_COMPLETE`。这些文件与逐 PDB NPZ 共同构成已经验收的 V3 几何池；其中 `config.json::entry_ratio` 和原 `validation_selection.npz` 记录历史 `0:5:5`/`0:1:1` 请求规则，不再决定新训练的样本集合。

当前活动训练直接复用上述逐 PDB候选数组，并由 Dataset 配置提供三个参数：

- `pdb_foreground_box_num=25`：每个 PDB 的目标 bias BOX 数量。
- `pdb_foreground_fraction_target=0.5`：bias BOX 占目标 bias 与 context BOX 总数的比例。
- `pdb_occurrence_foreground_box_cap=25`：单个 occurrence 在一个 epoch 内最多获得的 bias BOX 数量；该值等于每 PDB 的目标 bias 数量。

设一个 PDB 含 `O` 个 occurrence，实际 bias 数量为 `min(25, 25O)`。正式 pool 的每个 PDB 至少含一个 occurrence，因此 bias 数量固定为 25；bias 尽可能均匀地分给全部 occurrence，不能整除的余数沿稳定排列逐 epoch 轮转。context 数量同样固定为 25。正式 train 与 validation 的每个 PDB 都有超过 25 个 context 候选，选择时不需要放回或回退。

## PDB 中心验证选择的一次性冻结

`freeze_validation_selection_pdb_centric.py` 是保留在 `ops/stage1_data_preparation/` 中的硬编码生产脚本，不提供参数化命令行。脚本固定读取正式 validation pool，使用 `SeedSequence(3407, spawn_key=(2,))` 的独立随机域从 200 个 PDB 中无放回选择 150 个身份，再以 `25/0.5/25` 参数生成 epoch 0 请求并原子写入：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric.npz
```

准确运行命令只有一条：

```bash
python -m ops.stage1_data_preparation.freeze_validation_selection_pdb_centric
```

新 NPZ 精确包含以下 11 个字段：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `validation_pdb_id` | 定宽 bytes `(P,)` | PDB 身份表；三个 `*_pdb_index` 字段索引其第一维 |
| `center_pdb_index` | `int32 (0,)` | center 请求所属 PDB；PDB 中心规则下为空 |
| `center_occurrence_id` | `int32 (0,)` | center 请求所属 occurrence；PDB 中心规则下为空 |
| `bias_pdb_index` | `int32 (N_bias,)` | bias 请求所属 PDB |
| `bias_occurrence_id` | `int32 (N_bias,)` | bias 请求对应的真实配体 occurrence |
| `bias_candidate_index` | `int16 (N_bias,)` | 对应 occurrence 的 `bias_start_zyx` 候选编号 |
| `context_pdb_index` | `int32 (N_context,)` | context 请求所属 PDB |
| `context_candidate_index` | `int32 (N_context,)` | 对应 PDB 的 `context_start_zyx` 候选编号 |
| `pdb_foreground_box_num` | `int32` 标量 | 冻结时每个 PDB 的目标 bias BOX 数量，值为 25 |
| `pdb_foreground_fraction_target` | `float64` 标量 | 冻结时 bias 占目标总 BOX 的比例，值为 0.5 |
| `pdb_occurrence_foreground_box_cap` | `int32` 标量 | 冻结时单 occurrence 每个 epoch 的 bias 上限，值为 25 |

脚本不会修改逐 PDB NPZ、`manifest.json`、`validation_selection.npz`、`config.json`、`summary.json` 或 `_COMPLETE`。

## 2026-08-17 正式运行结果

- 数组迁移：22,381 个密度目录，89,524 个目标位置；89,442 个来源字段成功迁移，82 个目标位置没有来源文件。新发布 NPY 共 13,813,200,929,408 字节。
- 冻结划分：train 13,717 PDB，validation 200 PDB，calibration 100 PDB，日期留出 2,497 PDB，缺日期隔离 357 PDB。
- BOX pool：train 和 validation 分别发布 13,717 与 200 个 PDB NPZ；两者均无零 context PDB。2026-08-17 初次发布的验证选择包含 16,525 个 bias 与 16,525 个 context；2026-08-18 按历史 `0:1:1` 规则覆盖后的 `validation_selection.npz` 包含 3,305 个 bias 与 3,305 个 context。两份历史结果的 center 都为 0。
- PDB 中心验证选择：2026-08-24 从既有 200 个 validation PDB 中按 `SeedSequence(3407, spawn_key=(2,))` 无放回冻结 150 个身份，以及 3,750 个 bias、3,750 个 context 和 0 个 center 请求。150 个 PDB 都恰好包含 25 个 bias 与 25 个 context，并按原 validation manifest 顺序保存。`validation_selection_pdb_centric.npz` 为 71,150 字节，SHA-256 为 `546ebd3a1f07b230af42911b6740f466c6af289c8bff91a387c4eb8b1d69dd8e`。
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

第三版产物现在由唯一的 `src/datasets/stage1_dataset.py::Stage1Dataset` 直接消费。四个完整体 NPY 以只读 mmap 延迟打开；训练按 PDB 动态生成 `25` 个目标 bias 与 `25` 个固定 context 请求，验证展开 `validation_selection_pdb_centric.npz`。`utils/box_pool.py` 只供原 V3 构建入口复用，集中 occurrence mask、确定性 PDB seed、bias/context 起点和历史 validation selection 冻结逻辑；当前 PDB 中心选择由正式请求模块 `src/datasets/stage1_requests.py` 定义，一次性脚本只调用该生产逻辑并保存 epoch 0。
