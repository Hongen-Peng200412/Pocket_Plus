# Stage1 V3 Dataset 数据契约

本文说明 `Stage1Dataset` 当前正式训练使用的 V3 数据位置、请求比例和运行时裁块边界。历史版本的 split、BOX pool 与比例抽样实现只通过 Git 阅读，不再属于活动代码。

## 正式位置

完整图资产根目录为：

```text
/storage/penghongen/AdaLigand/Ori_Data
```

V3 split 与 BOX pool 根目录为：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3
├── split/
│   ├── train.json
│   ├── validation.json
│   ├── calibration.json
│   ├── held_out.json
│   ├── quarantine_missing_release.json
│   └── _COMPLETE
└── box_pool/
    ├── train/{pdb_id}.npz
    ├── validation/{pdb_id}.npz
    ├── manifest.json
    ├── validation_selection.npz
    ├── config.json
    └── _COMPLETE
```

`configs/dataset/stage1_find.yaml` 与 `stage1_unet_c1.yaml` 都指向上述位置。训练不扫描目录中的额外 PDB 文件，只读取 `manifest.json` 声明的文件。

## 完整图资产

每个 PDB 的四个大数组已经从压缩 NPZ 迁移为可内存映射的 NPY：

| 文件 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `density/{pdb_id}/exp.npy` | `float32 (1,D,H,W)` | 实验密度，空间轴为 ZYX |
| `density/{pdb_id}/sim.npy` | `float32 (1,D,H,W)` | 模拟密度，空间轴为 ZYX |
| `density/{pdb_id}/union_mask.npy` | `bool (1,D,H,W)` | 全部 occurrence 配体区域并集 |
| `density/{pdb_id}/ligand_dist.npy` | `float16 (1,D,H,W)` | 最近配体原子距离，单位 Å |

同名 NPZ 只保存几何、schema 和逐 occurrence 稀疏数据。`receptor_tokens.npz` 保存 `coords: float32 (N,3)`、`feat: float32 (N,49)` 与 `is_backbone: bool (N,)`；Dataset 返回 49 维特征和独立布尔字段，模型边界按模型输入维数决定是否拼成第 50 维。

Dataset 以 `numpy.load(..., mmap_mode="r")` 打开完整图，只复制实际 80³ 裁块。文件级检查只读取 NPY 头和小型 NPZ 元数据；有限值、距离非负等数值检查在实际裁块上执行。

## BOX pool 与请求

每个 PDB pool 保存：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `occurrence_id` | `int32 (O,)` | occurrence 身份 |
| `center_start_zyx` | `int32 (O,3)` | 兼容保留的中心起点；V3 正式请求数为 0 |
| `bias_start_zyx` | `int32 (O,30,3)` | 每个 occurrence 的 30 个 bias 候选 |
| `context_start_zyx` | `int32 (C,3)` | PDB 级 context 候选 |

所有起点都是完整图内 80³ BOX 的零基 ZYX corner index。训练每个 epoch、每个 PDB 至多选择 50 个 occurrence，并对每个 occurrence 选择 5 个 bias 与 5 个 context，请求比例固定为 `center:bias:context = 0:5:5`。验证直接展开 `validation_selection.npz`，不重新随机选择。

活动代码没有 `box_sample_fraction`，也不创建训练或验证比例请求文件。

## DataLoader 边界

- 单卡任务申请 16 CPU，并使用 16 个 DataLoader worker。
- 双卡 DDP 任务总共申请 32 CPU，每个 rank 使用 16 个 worker，总计 32 个 worker。
- `prefetch_factor=4`、`pin_memory=true`、`persistent_workers=false`。
- `persistent_workers=false` 是请求语义的一部分：主进程调用 `set_epoch` 后，新 worker 才能看到该 epoch 的请求序列。

## 代码入口

- `stage1_requests.py`：解析 manifest、单 PDB pool、动态训练请求和冻结验证请求。
- `stage1_dataset.py`：内存映射完整图、现场裁出 80³ BOX、构造监督与受体原子表。
- `stage1_collate.py`：堆叠体素张量并拼接变长受体原子字段。
- `src/model/stage1_model.py`：在模型边界处理 49 维基础特征与独立主链标志。

V3 数据准备、一次性迁移记录和构建命令见 `ops/stage1_data_preparation/README.md` 与 `EXECUTION.md`。
