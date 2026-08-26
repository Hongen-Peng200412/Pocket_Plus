# AdaLigand Stage1 Dataset 阅读入口

当前生产 Dataset 只有 `src/datasets/stage1_dataset.py::Stage1Dataset`。训练和后续推理共用其 80³ 裁块、密度通道、监督构造与受体原子选择逻辑，不维护第二套 V3 Dataset。

## 当前链路

```text
stage1_preparation_box_pool_3/box_pool/manifest.json
    → train/{pdb_id}.npz
      或 validation_selection_pdb_centric.npz
      或 validation_selection_pdb_centric_v2.npz
    → ResolvedStage1Crop(pdb_id, box_start_zyx, role, occurrence_id)
    → Stage1Dataset 从 exp/sim/union_mask/ligand_dist NPY 裁出 80³
    → Stage1BatchCollator
    → Find_0、Find_1、unet_base、unet_c1 或 unet_diff
```

-- 8.26(日期与实验区分见global/)
训练使用 `pdb_foreground_box_num=25`、`pdb_foreground_fraction_target=0.5` 与 `pdb_occurrence_foreground_box_cap=25`。正式 pool 的每个 PDB 至少含一个 occurrence，因此每个 PDB 每个 epoch 固定使用 25 个 bias 和 25 个 context；bias 尽可能均匀分到全部 occurrence，余数分配逐 epoch 轮转。验证以 seed 3407 从 200 个 validation PDB 中无放回选择 150 个身份，冻结同规则 epoch 0，并由所有当前模型共用 `validation_selection_pdb_centric.npz`。原 `0:5:5`/`0:1:1` 请求、旧版 `1:5:3`、`box_sample_fraction`、`src/datasets/ops/stage1_split.py` 与 `stage1_box_pool.py` 均已退出活动消费链，只能从现存历史产物或 Git 历史阅读。

-- 8.27(日期与实验区分见global/)
Find、unet_base 与 unet_diff 继续使用 V1 `25/0.5/25` 契约：每个 PDB 每个 epoch 固定使用 25 个 bias 和 25 个 context，验证由 `validation_selection_pdb_centric.npz` 冻结 150 个 PDB、3,750 个 bias 和 3,750 个 context。unet_c1 使用 V2 `50/0.7575757575757576/1` 契约：每个 occurrence 最多贡献一个 bias，每个 PDB 固定使用 16 个 context，验证由 `validation_selection_pdb_centric_v2.npz` 冻结全部 200 个 PDB、3,305 个 bias 和 3,200 个 context。两个验证文件共存，V2 不覆盖 V1。原 `0:5:5`/`0:1:1` 请求、旧版 `1:5:3`、`box_sample_fraction`、`src/datasets/ops/stage1_split.py` 与 `stage1_box_pool.py` 均已退出活动消费链，只能从现存历史产物或 Git 历史阅读。

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
- `Find_1.sh` 的双卡资源为 64 CPU、每个 rank 24 workers；`unet_c1.sh` 的单卡资源为 32 CPU、30 workers；其余当前 Stage1 入口每个 rank 使用 16 workers。
