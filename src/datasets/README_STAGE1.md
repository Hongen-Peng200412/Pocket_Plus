# AdaLigand Stage1 Dataset 阅读入口

当前生产 Dataset 只有 `src/datasets/stage1_dataset.py::Stage1Dataset`。训练和后续推理共用其 80³ 裁块、密度通道、监督构造与受体原子选择逻辑，不维护第二套 V3 Dataset。

## 当前链路

```text
stage1_preparation_box_pool_3/box_pool/manifest.json
    → train/{pdb_id}.npz 或 validation_selection.npz
    → ResolvedStage1Crop(pdb_id, box_start_zyx, role, occurrence_id)
    → Stage1Dataset 从 exp/sim/union_mask/ligand_dist NPY 裁出 80³
    → Stage1BatchCollator
    → Find_0、Find_1 或 unet_c1
```

训练请求固定为 `0:5:5`：每个 epoch、每个 PDB 至多选择 50 个 occurrence，每个 occurrence 选择 5 个 bias 与 5 个 context。验证使用已经冻结的同口径请求。旧版 `1:5:3`、`box_sample_fraction`、`src/datasets/ops/stage1_split.py` 与 `stage1_box_pool.py` 均已退出活动代码，只能从 Git 历史阅读。

## 阅读顺序

1. `src/datasets/readme.md`：正式路径、文件字段、请求比例和 DataLoader 资源契约。
2. `ops/stage1_data_preparation/README.md`：NPY 一次性迁移、V3 split 与 BOX pool 的构建证据。
3. `src/datasets/stage1_requests.py`：请求身份、顺序和每 epoch 随机选择。
4. `src/datasets/stage1_dataset.py`：mmap 缓存、80³ 裁块、增强与监督字段。
5. `src/datasets/stage1_collate.py`：batch 字段和变长原子表拼接。

## 必须保持的边界

- 四个大数组来自同目录 NPY；NPZ 不再保存对应完整体数组。
- Dataset 返回 `atom_feat: float32 (N,49)` 与 `atom_is_backbone: bool (N,)`；模型需要 50 维时才在输入边界拼接。
- 完整图不得为容纳 80³ 而补零；V3 split 已排除三轴任一长度小于 80 的 PDB。
- DataLoader 必须使用 `persistent_workers=false`，否则 worker 会保留旧 epoch 的请求集。
- 正式训练资源为单卡 16 CPU/16 workers，双卡总计 32 CPU/32 workers。
