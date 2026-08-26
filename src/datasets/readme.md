# Stage1 V3 Dataset 数据契约

本文说明 `Stage1Dataset` 当前正式训练使用的 V3 数据位置、两套 PDB 中心采样规则和运行时裁块边界。V3 逐 PDB 几何候选池保持不变；V1 与 V2 验证文件独立共存，并分别服务采样方式二与采样方式三。

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
    ├── validation_selection_pdb_centric.npz
    ├── validation_selection_pdb_centric_v2.npz
    ├── config.json
    └── _COMPLETE
```

`configs/dataset/stage1_find.yaml`、`stage1_unet_base.yaml`、`stage1_unet_c1.yaml` 与 `stage1_unet_diff.yaml` 都指向上述位置。训练不扫描目录中的额外 PDB 文件，只读取 `manifest.json` 声明的文件。Find、unet_base 与 unet_diff 读取 V1 `validation_selection_pdb_centric.npz`；unet_c1 读取 V2 `validation_selection_pdb_centric_v2.npz`。原 `validation_selection.npz` 与 `config.json::entry_ratio` 仅记录 V3 几何池的历史构建规则。

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

所有起点都是完整图内 80³ BOX 的零基 ZYX corner index。训练配置用 `pdb_foreground_box_num`、`pdb_foreground_fraction_target` 与 `pdb_occurrence_foreground_box_cap` 三个参数定义请求；这里的 foreground 只表示 bias BOX，context BOX 仍沿用原名。

V1 使用 `25/0.5/25`：每个 PDB 固定抽取 25 个 bias 与 25 个 context，bias 尽可能均匀分给全部 occurrence，不能整除的余数逐 epoch 轮转。V2 使用 `50/0.7575757575757576/1`：一个含 `O` 个 occurrence 的 PDB 抽取 `min(O,50)` 个 bias，每个 occurrence 最多一个；context 目标数量由 `round(50 × (1 - 25/33) / (25/33))` 得到，固定为 16。两种规则都从既有候选中无放回抽取；正式 pool 的 context 候选数满足这两个目标数量。

V1 以 `SeedSequence(3407, spawn_key=(2,))` 的独立随机域从 200 个 validation PDB 中选择 150 个身份，并把方式二的 epoch 0 请求冻结到 `validation_selection_pdb_centric.npz`。V2 使用全部 200 个 validation PDB，并把方式三的 epoch 0 请求冻结到 `validation_selection_pdb_centric_v2.npz`。两个文件都精确包含以下 11 个字段：

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
| `pdb_foreground_box_num` | `int32` 标量 | 冻结时每个 PDB 的目标 bias BOX 数量；V1 为 25，V2 为 50 |
| `pdb_foreground_fraction_target` | `float64` 标量 | 冻结时 bias 占目标总 BOX 的比例；V1 为 0.5，V2 为 0.7575757575757576 |
| `pdb_occurrence_foreground_box_cap` | `int32` 标量 | 冻结时单 occurrence 每个 epoch 的 bias 上限；V1 为 25，V2 为 1 |

Dataset 按文件顺序完整展开这些请求，不在验证期间重新抽样。三个采样参数标量只记录冻结契约；请求展开函数不读取或校验它们。

活动代码没有 `box_sample_fraction`，也不为训练请求落盘额外选择文件。训练请求由 manifest 和当前 epoch 动态生成；验证只接受上述两个精确文件名，不搜索或展开其他 NPZ。

## DataLoader 边界

- `Find_1.sh` 的双卡任务申请 64 CPU，每个 rank 使用 24 个 DataLoader worker；`unet_c1.sh` 的单卡任务申请 32 CPU，使用 30 个 worker；其余当前 Stage1 入口每个 rank 使用 16 个 worker。
- `prefetch_factor=4`、`pin_memory=true`、`persistent_workers=false`。
- `persistent_workers=false` 是请求语义的一部分：主进程调用 `set_epoch` 后，新 worker 才能看到该 epoch 的请求序列。

## 代码入口

- `stage1_requests.py`：解析 manifest、单 PDB pool、动态训练请求和冻结验证请求。
- `stage1_dataset.py`：内存映射完整图、现场裁出 80³ BOX、构造监督与受体原子表。
- `stage1_collate.py`：堆叠体素张量并拼接变长受体原子字段。
- `src/model/stage1_model.py`：在模型边界处理 49 维基础特征与独立主链标志。

V3 数据准备、一次性迁移记录和构建命令见 `ops/stage1_data_preparation/README.md` 与 `EXECUTION.md`。
