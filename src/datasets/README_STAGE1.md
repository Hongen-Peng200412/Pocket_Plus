# AdaLigand Stage1 Dataset

本目录中的 `stage1_*` 模块把 A—G 整图资产统一物化为 80³ 的训练、validation、完整图滑窗和居中推理样本。科学定义与字段权威仍是 AdaLigand 的三份主规格；本文只说明代码入口和发布顺序。

## 1. 清单身份、生产者与当前正式路径

这里有三个不能混用的清单名称：

| 名称 | 身份 | 当前正式位置或状态 |
|---|---|---|
| `keep_list.jsonl` | A—G Stage G `filter` 模式的正式输出；每行是一个 `(pdb_id, candidate_id)` 配体 occurrence | 未生成：规范位置为 `/storage/penghongen/AdaLigand/Ori_Data/keep_list.jsonl`；本次正式 A—G run 只执行了 `analyze`，因此该文件没有生成 |
| `strict_filtered_keep_list.jsonl` | Stage1 inventory 根据 Stage G 分析结果应用严格 map 条件后得到的中间清单 | `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory/strict_filtered_keep_list.jsonl` |
| `final_keep_list.jsonl` | Stage1 inventory 再通过 Dataset 直接资产审计后的最终训练身份清单 | `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory/final_keep_list.jsonl` |

### 1.1 A—G 的 `keep_list.jsonl` 如何产生

A—G 的正式生产者不是本目录的 Stage1 模块，而是 AdaLigand 数据管线中的：

- CLI：`Data_Preprocessing/Ori_Data/adaligand_preprocessing/cli/stage_g.py`
- 核心函数：`Data_Preprocessing/Ori_Data/adaligand_preprocessing/stages/stage_g.py::run_stage_g`

命令入口是：

```bash
python -m adaligand_preprocessing.cli.stage_g \
  --root /storage/penghongen/AdaLigand/Ori_Data \
  --run_id adaligand_ag_20260711T154658 \
  --mode filter \
  --config /absolute/path/to/schema_v2_filter.json
```

必要时还要按同一个正式 Stage F release 传入受控失败 waiver 及其 SHA-256。`--mode filter` 必须有显式 schema v2 配置，不能让程序猜阈值。

该命令的内部调用关系是：

```text
cli/stage_g.py::main
    → stages/stage_g.py::run_stage_g(mode="filter")
    → load_filter_config()
    → apply_map_filter_config()
    → write_jsonl(root / "keep_list.jsonl", kept)
```

Stage G 的主要输入是：

- `raw/pair_list.jsonl`：待处理 PDB 配对清单；
- Stage D/E/F 的状态和产物；
- `reports/runs/{run_id}/stage_g_analysis/candidates.pending.jsonl`：分析模式生成的 occurrence 级质量记录；
- schema v2 过滤配置。

Stage G 先按 `pdb_id` 聚合，再判断 CC、分辨率和 occurrence 合格比例等 map-level 条件。一个 PDB 通过后，`keep_list.jsonl` 保留该 PDB 的全部 occurrence，而不是只保留其中 `pair_pass=true` 的行。

`--mode analyze` 与 `--mode filter` 的边界非常重要：

```bash
python -m adaligand_preprocessing.cli.stage_g \
  --root /storage/penghongen/AdaLigand/Ori_Data \
  --run_id adaligand_ag_20260711T154658 \
  --mode analyze
```

分析模式只写：

```text
reports/runs/adaligand_ag_20260711T154658/stage_g_analysis/candidates.pending.jsonl
reports/runs/adaligand_ag_20260711T154658/stage_g_analysis/quality_distribution.json
```

它明确不会写根目录的 `keep_list.jsonl`。当前正式 A—G 运行就是这个状态，所以不能因为 `/storage/penghongen/AdaLigand/Ori_Data/keep_list.jsonl` 不存在，就判断 A—G 产物缺失。

### 1.2 当前 Stage1 的 `final_keep_list.jsonl` 如何产生

当前 Stage1 正式准备没有直接消费一个已经发布的 A—G 根目录 `keep_list.jsonl`。它使用一个 run-scoped 临时脚本：

```text
Pocket_Plus/tmp/stage1_prepare_inventory.py
```

该脚本的生产函数是 `prepare_inventory()`。它做两次筛选：

1. 流式读取 A—G `candidates.pending.jsonl`，按 PDB 聚合 `cc_contour` 和 `map_resolution`，只保留 `cc_contour > 0.6` 且 `map_resolution < 7.0` 的 PDB，原样写出 `strict_filtered_keep_list.jsonl`。如果严格通过的 PDB 少于 `--expected-minimum-pdb-count 18000`，脚本会把全部候选 PDB 纳入资产审计并在摘要中标记回退；本次严格清单超过门槛，因此 `fallback=false`。
2. 对严格清单中的 PDB 检查 Stage1 Dataset 直接需要的资产：`parse/{pdb_id}/receptor_tokens.npz`、`density/{pdb_id}/exp.npz`、`density/{pdb_id}/sim.npz`、`density/{pdb_id}/ligand_area.npz` 和 `labels/{pdb_id}/atom_labels.npz`，并检查 ZIP/NPY 头、字段、shape、origin、voxel size 和 occurrence identity。通过审计的 PDB 原样写出 `final_keep_list.jsonl`。

本次正式运行由以下 sbatch 脚本承载：

```text
Pocket_Plus/tmp/adaligand_stage1_inventory_split.sbatch
```

其中真正执行 inventory 的命令是：

```bash
python /home/penghongen/My_Project/Pocket_Plus/tmp/stage1_prepare_inventory.py \
  --candidates /storage/penghongen/AdaLigand/Ori_Data/reports/runs/adaligand_ag_20260711T154658/stage_g_analysis/candidates.pending.jsonl \
  --quality-distribution /storage/penghongen/AdaLigand/Ori_Data/reports/runs/adaligand_ag_20260711T154658/stage_g_analysis/quality_distribution.json \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --output-root /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory \
  --workers 48 \
  --expected-minimum-pdb-count 18000
```

该命令的直接输出是：

```text
inventory/strict_filtered_keep_list.jsonl
inventory/final_keep_list.jsonl
inventory/asset_audit.jsonl
inventory/summary.json
inventory/_COMPLETE
```

本次正式记录中的计数是：

```text
strict_filtered_keep_list.jsonl: 20,483 PDB，637,140 occurrence
final_keep_list.jsonl:           18,293 PDB，579,688 occurrence
```

2,190 个 PDB 被排除，唯一记录的原因是 `exp_sim_geometry_mismatch`，即 `exp.npz` 与 `sim.npz` 的 shape、voxel size 或 origin 不完全满足当前 Dataset 的一致性要求。

因此，当前训练使用的最终清单是 `final_keep_list.jsonl`，不是 A—G 根目录的 `keep_list.jsonl`。前者是 Stage1 为当前 Dataset 重新确认过资产兼容性的最终身份清单；后者是 A—G filter 模式的规范输出，两者不能当作同一个文件。

### 1.3 `final_keep_list.jsonl` 被谁消费

当前正式 preparation 的下一步由同一个 sbatch 脚本调用：

```text
Pocket_Plus/tmp/stage1_freeze_split_200.py
```

其实际命令是：

```bash
python /home/penghongen/My_Project/Pocket_Plus/tmp/stage1_freeze_split_200.py \
  --keep-list /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory/final_keep_list.jsonl \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --output-root /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/split \
  --seed 3407 \
  --validation-pdb-count 200 \
  --calibration-pdb-count 100
```

这个临时 wrapper 最终调用长期 Dataset API：

```text
src.datasets.stage1_split::freeze_stage1_splits()
```

该函数按 `pdb_id` 聚合 `final_keep_list.jsonl`，保证同一个 PDB 的全部 occurrence 不跨 split，然后写出：

```text
split/train.json
split/validation.json
split/calibration.json
split/held_out_pool.json
split/summary.json
split/_COMPLETE
```

此后 `final_keep_list.jsonl` 不再被训练 Dataset 直接读取。后续消费者是：

```text
final_keep_list.jsonl
    ↓ 直接消费
stage1_freeze_split_200.py / freeze_stage1_splits()
    ↓
split/{train,validation,calibration,held_out_pool}.json
    ↓ 直接消费
stage1_box_pool.py::build_pdb_box_pool()
    ↓
box_pool/manifest.json + validation_selection.npz + BOX NPZ
    ↓ 直接消费
stage1_requests.py
    ↓
stage1_dataset.py → stage1_collate.py → DataLoader → 训练
```

因此，训练期间 Dataset 读取的是 BOX pool、`manifest.json`、`validation_selection.npz` 以及 A—G 的密度/原子/标签资产，并不再次打开 `final_keep_list.jsonl`。

### 1.4 当前正式链路的最短复述

```text
A—G Stage F 质量产物
  → Stage G analyze
  → candidates.pending.jsonl
  → tmp/stage1_prepare_inventory.py
  → inventory/strict_filtered_keep_list.jsonl
  → Stage1 直接资产审计
  → inventory/final_keep_list.jsonl
  → tmp/stage1_freeze_split_200.py
  → split/*.json
  → src.datasets.stage1_box_pool
  → box_pool/manifest.json、validation_selection.npz 和 BOX 文件
  → stage1_requests.py
  → stage1_dataset.py
  → DataLoader 和模型训练
```

后续若只删除 `final_keep_list.jsonl` 中的 PDB，而不重新生成与之对应的 split、BOX pool manifest 和 validation selection，训练仍可能继续读取旧 BOX 池。因此清单、split 和 BOX pool 必须视为同一个已发布的 Stage1 preparation 版本，不能单独原地修改。

## 2. 冻结 split

长期通用入口可以直接使用 `src.datasets.stage1_split`；对当前正式 preparation，输入是上一节生成的 `inventory/final_keep_list.jsonl`，而不是假定存在的 A—G 根目录 `keep_list.jsonl`：

```bash
python -m src.datasets.stage1_split \
  --keep-list /absolute/path/to/stage1_preparation/inventory/final_keep_list.jsonl \
  --data-root /absolute/path/to/Ori_Data \
  --output-root /absolute/path/to/stage1_preparation/split
```

程序以 PDB 为不可跨 split 的分组键，固定 seed=3407；先从三轴均不小于 80 的 PDB 中冻结 validation 300 和 calibration 100，再取剩余 PDB 的前 `floor(0.75*N)` 个作为 train，其余进入尚未去冗余的 held-out pool。只有全部 JSON、配置和摘要写完才发布根 `_COMPLETE`。

## 2. 预计算 train/validation BOX pool

```bash
python -m src.datasets.stage1_box_pool \
  --data-root /absolute/path/to/Ori_Data \
  --train-split /absolute/path/to/stage1_preparation/split/train.json \
  --validation-split /absolute/path/to/stage1_preparation/split/validation.json \
  --output-root /absolute/path/to/stage1_preparation/box_pool
```

每个 occurrence 保存中心起点和 30 个球内均匀 bias 起点；每个 PDB 还保存 context 生成器实际找到的合法起点。池只有 1–2 项时有放回选满 3 项，池为空时省略 context 而不让 pool/训练失败。validation 的名义 1:5:3 选择冻结在 `validation_selection.npz`。发布清单 `manifest.json` 是 Dataset 的唯一 pool 索引，防止目录中遗留 NPZ 被静默读入。

## 3. 统一物化路径

- `stage1_requests.py` 提供 train、validation、完整图滑窗和 centered 请求。
- `stage1_dataset.py` 是唯一 materializer；每个 worker 通过带字节上限的缓存复用受体表和完整 exp/sim/union grid，避免同一 PDB 的窗口反复解压整图。
- `stage1_collate.py` 堆叠 dense voxel 字段，并用 `atom_offsets/atom_counts/atom_batch_index` 拼接变长原子表。
- Find 的 Dataset 直接加载 core+8 Å 原子；下游 artifact 中的 A-pocket 才按来源 blob 的 10 Å 包络与当前 BOX 取交集。
- train-only 90° 旋转会同步旋转 density/target/原子坐标；交换数组轴时也交换对应 voxel size，并重算 BOX 中心与世界坐标，不要求三轴尺度完全相等。

新版 Find_1 的监督样本还包含三张与 BOX 对齐的数组：

| 字段 | 形状与类型 | 内容 |
|---|---|---|
| `protein_mainchain_target` | `int64 (80,80,80)` | `0=背景, 1=N, 2=CA, 3=C, 4=O` |
| `nucleic_mainchain_target` | `int64 (80,80,80)` | `0=背景, 1=P, 2=O5', 3=C5', 4=C4', 5=C3', 6=O3'` |
| `ligand_inverse_distance_target` | `float32 (80,80,80)` | 从 `ligand_dist.npz` 裁出的 Å 距离按 `1/(1+d/(1 Å))` 变换；正无穷变为 0 |

蛋白和核酸类别只在受体原子局部坐标向下取整后所属的体素写入非背景编号，BOX 中其余体素是背景；损失在完整 `80³` 网格上计算。

训练请求在每个训练周期按 `occurrence_cap_per_pdb → occurrence_ratio → entry_ratio` 重新生成。当前正式配置依次为 `50 → 0.75 → 0:1:1`：每个 PDB 先无放回选择至多 50 个 occurrence，再保留 `ceil(一级候选数 × 0.75)` 个 occurrence，并各取一个 bias BOX 与一个 context BOX。第二版训练池 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2/box_pool` 当前包含 13,710 个 PDB；该配置在丢弃 DDP 尾部前固定产生 162,681 个 occurrence，即 325,362 个 BOX 请求。训练请求只存在于内存，不向 BOX 池写入选择文件；验证始终完整读取已有的 `validation_selection.npz`。

`Stage1PdbBatchSampler` 让同一 PDB 的请求在送入 DataLoader 的序列中连续，并按单个 DDP 进程一次前向使用的物理 `batch_size` 切分。DDP 表示分布式数据并行训练，rank 表示其中一个训练进程；center 和 bias BOX 属于前景，context BOX 属于背景。设当前 PDB 在当前物理 batch 中占据 `s` 个槽位、还剩 `F_left` 个前景 BOX 和 `N_left` 个全部 BOX，本片段使用 `floor(s × F_left / N_left + 1/2)` 个前景 BOX；剩余计数跨物理 batch 和由所有 rank 同时组成的 DDP 物理步继续使用。最后不足一个完整 DDP 物理步的请求直接丢弃，不补齐、不重复，也不产生缩小的 batch。首版关闭持久 DataLoader worker，避免 worker 保留上一训练周期的请求副本。

在 BOX 池、`request_seed`、训练周期、DDP 配置、物理 `batch_size`、worker 数和预取配置均相同时，请求身份与顺序可复现。修改物理 `batch_size` 或 worker 加载配置可能改变样本出现顺序。

训练入口仍是仓库根 `src/train.py`。AdaLigand 配置位于 `configs/experiment/CPC1/Find_0.yaml`、`Find_1.yaml`、`Find_2.yaml`、`configs/experiment/unet_c1.yaml` 及对应 Dataset、损失和训练子配置。合法 producer 统一由 `src/stage1_producers.py` 的 `STAGE1_MODEL_NAMES` 维护，其中 `FIND_MODEL_NAMES` 共享完整 56D density 和原子表物化语义。Find_2 的实现、配置与短训练证据继续保留，但当前正式训练不再使用 Find_2。
