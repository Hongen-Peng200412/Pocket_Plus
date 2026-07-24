# AdaLigand Stage1 Dataset

本目录中的 `stage1_*` 模块把 A—G 整图资产统一物化为 80³ 的训练、validation、完整图滑窗和居中推理样本。科学定义与字段权威仍是 AdaLigand 的三份主规格；本文只说明代码入口和发布顺序。

## 1. 冻结 split

必须等 Stage G 发布最终 `keep_list.jsonl` 后再执行：

```bash
python -m src.datasets.stage1_split \
  --keep-list /absolute/path/to/keep_list.jsonl \
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

训练入口仍是仓库根 `src/train.py`。AdaLigand 配置位于 `configs/experiment/CPC1/Find_0.yaml`、`Find_1.yaml`、`Find_2.yaml`、`configs/experiment/unet_c1.yaml` 及对应 Dataset/loss/train 子配置。当前合法 producer 统一由 `src/stage1_producers.py` 的 `STAGE1_MODEL_NAMES` 维护；其中 `FIND_MODEL_NAMES` 共享完整 56D density 和原子表物化语义。
