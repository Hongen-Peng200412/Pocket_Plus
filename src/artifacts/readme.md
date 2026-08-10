# Stage1 推理产物契约

本文说明 Pocket Plus 当前 Stage1 数据准备、完整图概率、阈值校准、组件森林、组件谱系组、Fα/Li/CLG/Selected 居中 BOX、Selector 打分与最终选择产物。读者不需要先阅读源代码，也应能据此判断：

- 一个产物由哪个命令生成；
- 文件位于哪里；
- 文件包含哪些字段；
- 每个字段的类型、形状、单位、空值和索引目标是什么；
- 哪些完成标记与跨文件关系必须成立；
- 消费者在读取前必须校验什么。

稳定的 Stage1 生产入口见 [`src/inference/README.md`](../inference/README.md)。本文只描述当前代码发布到磁盘的正式文件，不记录实现历史、临时文件名或某次运行的具体路径。

## 1. 范围、术语与共同规则

### 1.1 三个身份轴

- Stage1 模型来源：对应命令参数 `--producer`，只允许 `Find_0`、`Find_1`、`Find_2`、`unet_c1`。
- 数据划分：对应命令参数 `--split`，正式推理只允许 `calibration`、`validation`、`train`。
- PDB 身份：对应 `pdb_id`。推理入口把身份转为小写，并拒绝空身份和重复身份。

`Find_0`、`Find_1`、`Find_2` 使用体素模态 V、点模态 P 和原子模态 A；`unet_c1` 只使用体素模态 V。`occurrence` 是磁盘字段沿用的术语，表示一个真实配体实例。本文保留目录术语 `centered`，表示为一个来源组件解析合法 80³ BOX 并在该 BOX 中重新执行模型。本文中的“组件谱系组”保留字段缩写 CLG，指从一个合格的 `t_F1` 组件沿冻结阈值谱系扩展得到的一组候选组件。

`hardmask` 是受体占据位置的完整图布尔掩码；值为 `True` 的体素是受体原子所在位置。三个 Find 模型来源在完整图融合后把这些位置的配体概率置零。

### 1.2 文件格式

- JSON 文件使用 UTF-8。
- NPZ 文件不得包含 `object` dtype，必须能由 `numpy.load(path, allow_pickle=False)` 读取。
- 正式写入器先写临时文件，重读并校验字段后，再原子替换正式路径。
- NPZ 字段集合是精确集合：缺少字段或出现未声明字段都属于契约不一致；本文明确允许整组缺席的 Find P/A 字段除外。
- 文件存在不等于角色完成。消费者还必须检查完成标记和 PDB 级异常状态。

字段名中的 `offsets` 表示变长表边界数组。第 `i` 个对象对应半开区间 `[offsets[i],offsets[i+1])`；每个具体 offsets 切分哪些值表，会在相应文件字段表中逐一写明。

### 1.3 形状记号

| 记号 | 含义 |
| --- | --- |
| `D,H,W` | 完整图的 Z、Y、X 三轴长度 |
| `N_occ` | 当前 PDB 的真实配体 occurrence 数 |
| `N_node` | 组件森林节点数 |
| `N_CLG` | 组件谱系组数 |
| `N_candidate` | 全部组件谱系组包含的候选组件总数 |
| `N_entry` | centered NPZ 中的归档项数 |
| `N_success` | Selected 归档中 `refine_status == 0` 的归档项数 |
| `L_*` | 相应变长值表的第一维总长度 |
| `C_*` | checkpoint 实际产生的特征宽度，不由产物契约写死 |

本文把 NumPy 形状写成 `(N, 3)`，把 JSON 数组长度写成“长度 3”。

### 1.4 坐标与单位

- 数组空间轴和离散体素索引使用 ZYX 顺序。
- 世界坐标和连续局部坐标使用 XYZ 顺序，单位为 Å。
- `origin_xyz` 与 `box_origin_world` 都表示网格角点，不是第一个体素中心。
- `voxel_size_xyz` 与 `voxel_size_world` 使用 XYZ 顺序，单位为 Å/voxel。
- 完整体素中心满足：

  `world_xyz = origin_xyz + (index_xyz + 0.5) * voxel_size_xyz`

- BOX 局部连续坐标满足：

  `local_xyz = (world_xyz - box_origin_world) / voxel_size_xyz`

- BOX 角点满足：

  `box_origin_world = origin_xyz + box_start_xyz * voxel_size_xyz`

- 形状为 `80×80×80` 的 BOX 中心是 `box_origin_world + 40 * voxel_size_xyz`。
- 合法 BOX 起点逐轴满足 `0 <= box_start_axis <= full_shape_axis - 80`。正式裁剪不使用空间填充。

### 1.5 运行时引用、但不重复保存的完整图资产

以下文件位于命令参数 `--data-root` 指向的数据根目录。它们是 centered 产物的来源，不会完整复制到 centered NPZ。

| 路径 | 当前用途 |
| --- | --- |
| `density/{pdb_id}/exp.npz` | 所有 Stage1 模型来源使用的实验密度和完整图几何 |
| `density/{pdb_id}/sim.npz` | 三个 Find 模型来源额外使用的模拟密度；几何必须与实验密度一致 |
| `density/{pdb_id}/ligand_area.npz` | 配体区域并集、逐 occurrence 稀疏掩码、校准真值和候选组件交集真值 |
| `density/{pdb_id}/ligand_dist.npz` | 训练时转换为 `1 / (1 + distance_Å)` 的最近配体原子距离 |
| `parse/{pdb_id}/receptor_tokens.npz` | 受体原子世界坐标、49 维基础特征、残基类别和原子名 |
| `labels/{pdb_id}/atom_labels.npz` | 与完整受体原子表对齐的 `binding_atom` 标签 |

`exp.npz` 与 `sim.npz` 的 `grid` 是 `float32 (1,D,H,W)`，`voxel_size` 和 `origin` 都是 `float32 (3,)` XYZ 数组。`ligand_area.npz` 的 `union_mask` 是 `bool (1,D,H,W)`；`True` 表示至少一个真实配体实例占据该体素。每个 `mask_{occurrence_id}` 是按 ZYX 字典序排列且不重复的整数 `(K_occ,3)` 稀疏坐标表。`ligand_dist.npz` 的 `distance` 是 `float16 (1,D,H,W)`。`receptor_tokens.npz` 的 `coords` 是 `float32 (N_receptor,3)` 世界 XYZ 坐标，`feat` 是 `float32 (N_receptor,49)`；`binding_atom` 是与它们第一维对齐的 `bool (N_receptor,)`。

Find centered 产物中的 `A_global_index` 指回 `receptor_tokens.npz` 第一维，用于原子身份追踪；同一批被保留原子的 49 维原始特征另以 `A_feat_L0` 保存在 centered NPZ 中，Selector 不需要再次读取 `receptor_tokens.npz`。

### 1.6 80³ BOX 的运行时物化

BOX 池和 centered 几何都不保存实际密度裁剪。Dataset 或推理运行时使用 `pdb_id + box_start_zyx` 从完整图现场裁出精确 `80×80×80` 数组。

- 三个 Find 模型来源必须按 `ALL_CHANNEL_NAMES` 的固定顺序构造全部 56 个密度通道。顺序是基础运算 `exp`、`sim`、`diff`、`posdiff`，每种运算依次展开归一化方案 `nonorm`、`clipnorm`，再依次展开后处理 `nopost`、`gauss1`、`gauss2`、`DoG1`、`DoG2`、`smooth1`、`smooth2`。
- `unet_c1` 的密度输入必须精确为单通道 `exp_clipnorm_nopost`。
- Find 从完整受体表选择 80³ 核心及其外侧 8 Å 缓冲范围内的原子，并保存这些原子在完整受体表中的编号；`unet_c1` 不构造原子输入表。
- 需要监督时，Dataset 现场构造配体体素标签、配体区域并集、蛋白主链类别、核酸主链类别、配体反距离和受体原子结合标签；这些训练数组不写入 BOX 池或 centered 推理产物。

## 2. 产物与生产命令总览

| 产物 | 生产命令 |
| --- | --- |
| 冻结数据划分 | `python -m src.datasets.ops.stage1_split` |
| 训练预定位 BOX 池 | `python -m src.datasets.ops.stage1_box_pool` |
| calibration 完整图概率 | `python -m src.inference.cli cal-probability` |
| 模型来源级阈值和校准指标 | `python -m src.inference.cli freeze-thresholds` |
| calibration 组件与 F1 centered | `python -m src.inference.cli cal-produce-f1` |
| calibration 组件、F1 centered、CLG centered | `python -m src.inference.cli cal-produce-f1-clg` |
| validation 的概率、组件与 F1 centered | `python -m src.inference.cli val-produce-prob-f1` |
| validation 的概率、组件、F1 centered、CLG centered | `python -m src.inference.cli val-produce-prob-f1-clg` |
| train 的概率、组件与 F1 centered | `python -m src.inference.cli train-produce-prob-f1` |
| train 的概率、组件、F1 centered、CLG centered | `python -m src.inference.cli train-produce-prob-f1-clg` |
| Selector 冻结输入清单 | `python -m src.selector.train --config <selector.yaml>` 在训练启动时创建 |
| Selector 每 PDB 分数 | `python -m src.selector.inference scores` |
| Selector 门控阈值 | `python -m src.selector.inference calibrate` |
| Selector 每 PDB 选择表 | `python -m src.selector.inference selection` |
| Selected 精修 centered | `python -m src.inference.cli selected-refined` |

命令的完整参数、前置条件和执行顺序见 [`src/inference/README.md`](../inference/README.md)。下面按磁盘产物逐一给出契约。

## 3. 冻结数据准备产物

数据准备产物位于调用方传给 `--output-root` 的目录，不属于 `stage1_outputs`。

### 3.1 冻结数据划分

生产命令：

```text
python -m src.datasets.ops.stage1_split --keep-list <keep_list.jsonl> --data-root <数据根目录> --output-root <数据划分目录> --seed <整数>
```

输出目录：

```text
<数据划分目录>/
├── train.json
├── validation.json
├── calibration.json
├── held_out_pool.json
├── config.json
├── summary.json
└── _COMPLETE
```

`train.json`、`validation.json`、`calibration.json`、`held_out_pool.json` 都是 JSON 对象数组。每项原样保留输入 keep-list 中相应 PDB 的记录字段，并且至少包含字符串 `pdb_id`。同一个 PDB 不会跨数据划分。

选择规则：

- `validation` 和 `calibration` 只接收完整图 Z、Y、X 三轴都不小于 80 的 PDB。
- 默认选择 300 个 validation PDB 和 100 个 calibration PDB。
- 设输入中的唯一 PDB 总数为 `N`；train 从两个评估数据划分之外选择 `floor(0.75 * N)` 个 PDB，其余进入 `held_out_pool`。
- train 数据划分不预先排除短图；后续 BOX 池不会为三轴任一长度小于 80 的 train PDB 发布 PDB 级 NPZ，并在 `summary.json` 的 `short_map_count` 中计数。
- `held_out_pool` 不执行额外去重或形状过滤。
- 当前实现不创建独立的 `eligibility` 目录、合格或排除 PDB 清单，也没有对应生产命令。短图计数只出现在普通 `summary.json` 中。

命令参数 `--seed` 的默认值是 3407。`config.json`：

| 字段 | 类型与值 |
| --- | --- |
| `schema_version` | `int`，当前为 `1` |
| `seed` | `int`，实际使用的排名种子 |
| `train_fraction` | `float`，当前为 `0.75` |
| `validation_pdb_count` | `int`，请求的 validation PDB 数 |
| `calibration_pdb_count` | `int`，请求的 calibration PDB 数 |
| `minimum_validation_calibration_shape_zyx` | 长度 3 的 `int` 数组，当前为 `[80,80,80]` |
| `assignment` | `str`，当前为 `"sha256(seed|purpose|pdb_id) ascending"` |
| `keep_list_path` | `str`，调用方传入的 keep-list 路径 |
| `keep_list_sha256` | `str`，输入文件的 64 个十六进制字符 SHA-256 文本 |
| `held_out_deduplication` | `bool`，当前为 `false` |

`summary.json`：

| 字段 | 类型与含义 |
| --- | --- |
| `seed` | `int`，实际使用的排名种子 |
| `source_pdb_count` | `int`，输入中的唯一 PDB 数 |
| `source_row_count` | `int`，输入 keep-list 的原始记录数 |
| `validation_calibration_shape_checked_pdb_count` | `int`，评估数据划分选择期间检查过形状的 PDB 数 |
| `short_map_encountered_while_selecting_eval_count` | `int`，评估数据划分选择期间遇到的短图 PDB 数 |
| `splits` | JSON 对象，键为四个数据划分名 |
| `splits[name].pdb_count` | `int`，相应数据划分的唯一 PDB 数 |
| `splits[name].row_count` | `int`，相应数据划分保留的原始记录数 |

`_COMPLETE` 是零字节文件，在上述所有文件发布成功后最后创建。

### 3.2 训练预定位 BOX 池

生产命令：

```text
python -m src.datasets.ops.stage1_box_pool --data-root <数据根目录> --train-split <train.json> --validation-split <validation.json> --output-root <BOX池目录> --seed <整数>
```

输出目录：

```text
<BOX池目录>/
├── train/{pdb_id}.npz
├── validation/{pdb_id}.npz
├── manifest.json
├── validation_selection.npz
├── config.json
├── summary.json
└── _COMPLETE
```

每个 `train/{pdb_id}.npz` 或 `validation/{pdb_id}.npz`：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `pdb_id` | Unicode 标量 | 当前文件的 PDB 身份 |
| `occurrence_id` | `int32 (N_occ,)` | 当前 PDB 的正式 occurrence 编号 |
| `center_start_zyx` | `int32 (N_occ,3)` | 与 `occurrence_id` 同序的居中正样本 BOX 起点 |
| `bias_start_zyx` | `int32 (N_occ,30,3)` | 每个 occurrence 的 30 个偏置正样本 BOX 起点 |
| `context_start_zyx` | `int32 (N_context,3)` | 与 occurrence 无关的受体上下文 BOX 起点 |

上下文 BOX 从逐轴合法的整数起点均匀采样。80³ 核心中至少包含 1000 个受体重原子才保留；生成器不按配体位置过滤。每个 PDB 最多保留 500 个上下文 BOX，最多尝试 3000 次，因此 `N_context` 可以是 0 到 500。

不同 bias 随机样本解析到同一合法整数 BOX 起点时，重复起点原样保留。

`manifest.json`：

- `schema_version`：`int`，当前为 `1`。
- `splits`：JSON 对象，只含 `train` 和 `validation`。
- `splits[name]`：对象数组；每项只有 `pdb_id: str` 与 `path: str`。
- `path`：相对于 BOX 池根目录的 POSIX 风格路径，例如 `train/1abc.npz`。

`validation_selection.npz` 冻结一次 `center:bias:context = 1:5:3` 的验证请求：

| 字段 | dtype 与形状 | 索引目标 |
| --- | --- | --- |
| `validation_pdb_id` | 固定宽度 bytes `(N_pdb,)` | validation PDB 身份表 |
| `center_pdb_index` | `int32 (N_center,)` | 索引 `validation_pdb_id` 第一维 |
| `center_occurrence_id` | `int32 (N_center,)` | 在相应 PDB 的 `occurrence_id` 中查找同值 |
| `bias_pdb_index` | `int32 (N_bias,)` | 索引 `validation_pdb_id` 第一维 |
| `bias_occurrence_id` | `int32 (N_bias,)` | 在相应 PDB 的 `occurrence_id` 中查找同值 |
| `bias_candidate_index` | `int16 (N_bias,)` | 索引相应 occurrence 的 `bias_start_zyx` 第二维，范围 `0..29` |
| `context_pdb_index` | `int32 (N_context_selected,)` | 索引 `validation_pdb_id` 第一维 |
| `context_candidate_index` | `int32 (N_context_selected,)` | 索引相应 PDB 的 `context_start_zyx` 第一维 |

每个 PDB 每次最多选择 50 个 occurrence。上下文池有 1 或 2 项时允许放回采样至 3 项；上下文池为空时不伪造上下文请求。

validation 使用冻结请求，不保存增强后的数组。train 的随机 90° 旋转会同步旋转密度、监督图和 Find 原子坐标；奇数次四分之一转交换空间轴时，还会交换 `voxel_size_world` 的对应 XYZ 尺度，并重新计算 BOX 中心和原子世界坐标，不能用“体素尺寸近似 1 Å”代替几何变换。

命令参数 `--seed` 的默认值是 3407。`config.json`：

| 字段 | 类型与值 |
| --- | --- |
| `box_shape_zyx` | 长度 3 的 `int` 数组，当前为 `[80,80,80]` |
| `bias_candidates_per_occurrence` | `int`，当前为 `30` |
| `bias_radius_formula` | `str`，当前为 `"R=(3*K_occ/(4*pi))**(1/3)"` |
| `bias_selected_per_epoch` | `int`，当前为 `5` |
| `context_generator.sampling` | `str`，当前为 `"uniform_integer_legal_start_per_axis"` |
| `context_generator.target_count` | `int`，当前为 `500` |
| `context_generator.max_attempts` | `int`，当前为 `3000` |
| `context_generator.min_core_receptor_heavy_atoms` | `int`，当前为 `1000` |
| `context_generator.ligand_filter` | `bool`，当前为 `false` |
| `occurrence_cap_per_pdb_per_epoch` | `int`，当前为 `50` |
| `entry_ratio` | 对象，当前为 `{"center":1,"bias":5,"context":3}` |
| `train_random_rotation_90_degree` | `bool`，当前为 `true` |
| `seed` | `int`，BOX 池基准种子 |
| `seed_rule` | `str`，当前为 `"sha256(base_seed|split_name|pdb_id) first_uint64"` |

`summary.json`：

- 根字段 `seed` 是 `int`。
- `train` 包含 `requested_pdb`、`published_pdb`、`short_map_count`、`zero_context_pdb_count`、`underfilled_context_pdb_count`，全部为 `int`。
- `validation` 包含 `requested_pdb`、`published_pdb`、`zero_context_pdb_count`、`underfilled_context_pdb_count`，全部为 `int`。
- `validation_selection` 包含 `pdb_count`、`center_count`、`bias_count`、`context_count`，全部为 `int`。
- `manifest` 包含 `train` 和 `validation` 两个 `int` 文件计数。

`_COMPLETE` 是零字节文件，在上述所有文件发布成功后最后创建。

### 3.3 训练请求不形成落盘产物

训练请求由 `Stage1TrainingRequestSet` 在每个训练周期根据 PDB BOX 池、`request_seed`、`occurrence_cap_per_pdb`、`occurrence_ratio` 和 `entry_ratio` 在内存中生成。请求层不发布训练比例表，也不在 BOX 池目录中写入派生请求文件。

`validation_selection.npz` 仍是主仓库验证请求的固定入口。训练采样字段不会缩减、替换或重写该文件及其引用的 validation PDB NPZ。

## 4. Stage1 正式输出目录与状态

### 4.1 固定目录

对 `src.inference.cli`，命令参数 `--output-root` 指向下图中的 `<stage1_outputs>`：

```text
<stage1_outputs>/
└── {producer}/
    ├── calibration/
    │   ├── thresholds.json
    │   ├── threshold_scan.npz
    │   ├── metrics.json
    │   └── _COMPLETE
    └── {split}/{pdb_id}/
        ├── _RUNNING/owner.json
        ├── _BLOB_EXCEED
        ├── status/
        │   ├── probability/_COMPLETE
        │   ├── components/_COMPLETE
        │   ├── F_1_2_centered/_COMPLETE
        │   ├── F_2_3_centered/_COMPLETE
        │   ├── F_4_5_centered/_COMPLETE
        │   ├── F1_centered/_COMPLETE
        │   ├── F_5_4_centered/_COMPLETE
        │   ├── F_3_2_centered/_COMPLETE
        │   ├── F_2_centered/_COMPLETE
        │   ├── Li_centered/_COMPLETE
        │   ├── CLG_centered/_COMPLETE
        │   └── Selected_Refined_Centered/_COMPLETE
        ├── probability/
        │   ├── probability_map.npz
        │   └── geometry.json
        ├── components/
        │   ├── forest.npz
        │   ├── clg.npz
        │   ├── overlap.npz
        │   └── summary.json
        ├── centered/
        │   ├── F_1_2_centered.npz
        │   ├── F_2_3_centered.npz
        │   ├── F_4_5_centered.npz
        │   ├── F1_centered.npz
        │   ├── F_5_4_centered.npz
        │   ├── F_3_2_centered.npz
        │   ├── F_2_centered.npz
        │   ├── Li_centered.npz
        │   ├── CLG_centered.npz
        │   └── Selected_Refined_Centered.npz
        └── selector/selection.npz
```

`selector/selection.npz` 是 `selected-refined` 未传 `--selection-root` 时的默认输入位置；Stage1 概率、组件和居中归档生产命令不会自动创建它。

### 4.2 `_RUNNING/owner.json`

推理入口开始处理一个 PDB 时，通过原子创建 `_RUNNING` 目录取得 PDB 级租约。`owner.json` 的精确字段为：

| 字段 | 类型与含义 |
| --- | --- |
| `owner_token` | `str`，本次租约身份 |
| `pid` | `int`，创建租约的进程号 |
| `host` | `str`，创建租约的主机名 |
| `created_at_utc` | `str`，ISO 格式 UTC 时间 |

代码不会自行判断残留租约是否陈旧。发现已有 `_RUNNING` 时，当前 PDB 返回 `skipped_running`；只能在外部确认原进程已经终止后清理。

### 4.3 角色完成标记

PDB 级正式角色包括：

1. `probability`
2. `components`
3. 七个 Fα centered 角色：`F_1_2_centered`、`F_2_3_centered`、`F_4_5_centered`、`F1_centered`、`F_5_4_centered`、`F_3_2_centered`、`F_2_centered`
4. 独立 Li 变体的 `Li_centered`
5. `CLG_centered`
6. `Selected_Refined_Centered`

每个 `status/{role}/_COMPLETE` 是 JSON 对象，精确字段为：

| 字段 | 类型与含义 |
| --- | --- |
| `output_role` | `str`，必须等于目录中的角色名 |
| `completed_at_utc` | `str`，ISO 格式 UTC 完成时间 |

角色载荷通过 schema 重读校验并发布后，才发布该角色的 `_COMPLETE`。已有完整角色再次运行时返回 `skipped_complete`。

### 4.4 `_BLOB_EXCEED`

组件阶段统计的 `N_F1_eligible` 大于本次命令的 `--f1-eligible-limit` 时，发布 `_BLOB_EXCEED` JSON：

| 字段 | 类型与含义 |
| --- | --- |
| `N_F1_eligible` | `int`，当前 PDB 在 `t_F1` 层合格的组件数 |
| `limit` | `int`，本次命令实际使用的上限 |

默认行为仍在标记后停止后续生产。显式启用 `continue_on_blob_exceed` 时，标记保留，但请求的 components 与 centered 角色继续按原契约发布；该开关只改变是否继续生产，不改变任何数组字段。评估使用独立的 `evaluate_on_blob_exceed` 开关：开启后，只要该评估所需产物完整，就纳入指标；关闭时仍排除带标记的 PDB。生产开关与评估开关不能互相推导。

PDB 处理汇总状态只有 `completed`、`skipped_complete`、`skipped_running`、`blob_exceed`。

## 5. 完整图概率产物

### 5.1 生产入口与融合规则

`calibration` 的 probability 由 `cal-probability` 生成；`validation` 的 probability 由 `val-produce-prob-f1` 或 `val-produce-prob-f1-clg` 生成，`train` 的 probability 由 `train-produce-prob-f1` 或 `train-produce-prob-f1-clg` 生成。

- 滑窗形状为 `80×80×80`，步长为 `40×40×40`，不使用空间填充。
- 每个轴补入最后一个合法起点，使末端被最后一个窗口覆盖。
- 窗口按 Z、Y、X 的笛卡尔积确定性枚举。
- 融合权重是在 `[-1,1]^3` 上构造的归一化高斯权重，`sigma=0.5`。
- 概率加权和与权重和都使用 `float32`。
- 三个 Find 模型来源的概率在写盘前已乘 hardmask；`unet_c1` 保留模型概率。
- 窗口起点和融合过程的 `weight_sum` 只存在于内存，不属于磁盘产物。

### 5.2 `probability/probability_map.npz`

精确字段只有：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `probability_map` | `float32 (D,H,W)` | ZYX 完整图上的融合配体概率 |
| `origin_xyz` | `float32 (3,)` | 完整网格角点的世界 XYZ 坐标，单位 Å；与 `geometry.json` 的同名字段逐值一致 |
| `voxel_size_xyz` | `float32 (3,)` | 世界 XYZ 三轴的体素尺寸，单位 Å/voxel；三个值均为正，并与 `geometry.json` 的同名字段逐值一致 |

三个字段的数值都必须有限。`probability_map.npz` 因此可以单独支持概率分析、体素索引到世界坐标的换算和可视化；`geometry.json` 继续保存窗口、步幅与高斯参数等完整运行元数据。

### 5.3 `probability/geometry.json`

| 字段 | 类型、形状与含义 |
| --- | --- |
| `full_shape_zyx` | 长度 3 的 `int` 数组，必须等于 `probability_map.shape` |
| `origin_xyz` | 长度 3 的 `float` 数组，完整网格角点世界坐标，单位 Å；转换为 `float32` 后必须与 `probability_map.npz` 的同名字段逐值一致 |
| `voxel_size_xyz` | 长度 3 的正 `float` 数组，XYZ 体素尺寸，单位 Å/voxel；转换为 `float32` 后必须与 `probability_map.npz` 的同名字段逐值一致 |
| `window_shape_zyx` | 长度 3 的 `int` 数组，固定为 `[80,80,80]` |
| `stride_zyx` | 长度 3 的 `int` 数组，固定为 `[40,40,40]` |
| `gaussian_sigma` | `float`，固定为 `0.5` |

## 6. 模型来源级校准产物

生产命令：

```text
python -m src.inference.cli freeze-thresholds --producer <模型来源> --pdb-list <calibration清单> --data-root <数据根目录> --output-root <stage1_outputs> --min-voxels <整数> --max-voxels <整数> --denominator <整数>
```

该命令读取完整 calibration 清单中已完成的 probability 和真实标签，不读取 checkpoint。任一清单 PDB 的 probability 不可消费时，不得用部分 PDB 冻结校准。

概率 `p` 的离散阈值编号是 `floor(p * denominator)` 裁剪到 `[0, denominator]`。alpha 顺序固定为 `[0.5, 2/3, 0.8, 1.0, 1.25, 1.5, 2.0]`；并列最优时取从低到高扫描中首次出现的最大值。`t_F1` 是 alpha 等于 1 的冻结阈值。三维组件连通性固定为 26 邻域。

### 6.1 `calibration/thresholds.json`

| 字段 | 类型、形状与含义 |
| --- | --- |
| `stage1_model_name` | `str`，模型来源 |
| `denominator` | `int`，阈值离散分母 |
| `alpha_values` | 长度 7 的 `float` 数组，固定 alpha 顺序 |
| `alpha_threshold_grid_index` | 长度 7 的 `int` 数组，各 alpha 的冻结阈值编号 |
| `t_alpha` | 长度 7 的 `float` 数组，各阈值编号除以 `denominator` 的结果 |
| `t_F1` | `float`，alpha 等于 1 的阈值 |
| `min_voxels` | `int`，正式候选组件体素数下限，包含端点 |
| `max_voxels` | `int`，正式候选组件体素数上限，包含端点 |
| `connectivity` | `int`，固定为 `26` |

默认 `denominator=32768`、`min_voxels=10`、`max_voxels=2046`，但消费者必须读取实际文件值；calibration、validation 和 train 的组件与 F1 居中阶段必须共享同一份冻结值。

### 6.2 `calibration/threshold_scan.npz`

| 字段 | dtype 与形状 | 对齐关系 |
| --- | --- | --- |
| `denominator` | `int32` 标量 | 与 `thresholds.json` 同值 |
| `alpha_values` | `float64 (7,)` | 与 `thresholds.json` 同序同值 |
| `f_alpha_curve` | `float64 (7,denominator+1)` | 第一维索引 alpha，第二维索引阈值编号 |
| `tp` | `int64 (denominator+1,)` | 每个阈值编号的完整图体素 TP |
| `fp` | `int64 (denominator+1,)` | 每个阈值编号的完整图体素 FP |
| `fn` | `int64 (denominator+1,)` | 每个阈值编号的完整图体素 FN |

### 6.3 `calibration/metrics.json`

固定元数据：

| 字段 | 类型与含义 |
| --- | --- |
| `stage1_model_name` | `str`，模型来源 |
| `result_scope` | `str`，固定为 `"calibration_fitted"` |
| `threshold_scan` | `str`，固定指向 `threshold_scan.npz` 文件名 |

体素级字段：

| 字段 | 类型与含义 |
| --- | --- |
| `voxel_average_precision_macro` | `float`，有效 PDB 的平均精确率等权均值；没有有效 PDB 时为 NaN |
| `n_valid_voxel_ap_pdb` | `int`，至少含一个真实正体素的 PDB 数 |
| `n_total_pdb` | `int`，calibration 清单 PDB 总数 |
| `n_evaluated_pdb` | `int`，实际进入全部拟合评估指标的 PDB 数；是否纳入超限 PDB 由 `evaluate_on_blob_exceed` 决定 |
| `semantic_dice_micro_t_F1` | `float`，先汇总全部未超限 PDB 在 `t_F1` 上的 TP、FP、FN，再按 `2TP/(2TP+FP+FN)` 计算；总分母为 0 时为 `0.0` |
| `semantic_dice_macro_t_F1` | `float`，每个未超限 PDB 分别在 `t_F1` 上计算 Dice 后等权平均；单个 PDB 的分母为 0 时，该 PDB 的 Dice 按 `0.0` 进入平均 |
| `semantic_tp_t_F1` | `int`，全部未超限 PDB 在 `t_F1` 上汇总的 TP，属于 micro 聚合计数 |
| `semantic_fp_t_F1` | `int`，全部未超限 PDB 在 `t_F1` 上汇总的 FP，属于 micro 聚合计数 |
| `semantic_fn_t_F1` | `int`，全部未超限 PDB 在 `t_F1` 上汇总的 FN，属于 micro 聚合计数 |
| `n_blob_exceed_pdb` | `int`，冻结 `t_F1` 后合格组件数大于固定统计界限 200、因而从平均精确率、Dice、实例与 top-K 指标中完全排除的 PDB 数 |
| `evaluate_on_blob_exceed` | `bool`，本次 fitted 指标是否纳入具有可用产物的超限 PDB |

阈值扫描阶段仍使用 calibration 清单中全部可读概率图选择阈值。冻结后才构建单层组件；`evaluate_on_blob_exceed=false` 时超限 PDB 不以零分代替且不进入指标分母，`true` 时只要产物完整便照常纳入。

实例级字段：

- `n_pred_instances: int`、`n_gt_instances: int`。
- 对 `tag` 为 `0p3` 和 `0p5`，分别保存 `coverage_precision_{tag}`、`coverage_recall_{tag}`、`coverage_f1_{tag}`、`one_to_one_precision_{tag}`、`one_to_one_recall_{tag}`、`one_to_one_f1_{tag}`，类型均为 `float`。字段前缀 `coverage` 表示双向覆盖匹配，`one_to_one` 表示固定一对一匹配。
- `n_topk_eligible_pdb: int`。
- 对 `K` 为 3、4、5，且 `tag` 为 `0p3`、`0p5`，保存 `top{K}_success_{tag}: int` 与 `top{K}_success_ratio_{tag}: float`；它们统计按组件平均概率排序的前 K 个预测中是否出现达标交集。

### 6.4 `calibration/_COMPLETE`

该文件是 JSON 对象：

| 字段 | 类型与含义 |
| --- | --- |
| `stage1_model_name` | `str`，模型来源 |
| `result_scope` | `str`，固定为 `"calibration_fitted"` |

它在 `thresholds.json`、`threshold_scan.npz` 和 `metrics.json` 都成功发布后最后创建。

## 7. 组件森林与组件谱系组产物

六个 F1 或 F1/CLG 生产命令都会产生本节文件。calibration 命令复用既有 probability；validation 和 train 命令先补齐 probability。六个命令都要求模型来源级校准已经完整发布。

### 7.1 `components/forest.npz`

主节点表按 `(tree_id, node_id)` 升序排列，二者共同构成全局节点身份。每棵树的 `node_id` 从 0 开始连续编号。父节点位于相邻的较低阈值层，子节点位于相邻的较高阈值层。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `tree_id` | `int32 (N_node,)` | 组件树编号 |
| `node_id` | `int32 (N_node,)` | 当前树内节点编号 |
| `threshold_grid_index` | `int32 (N_node,)` | 阈值编号 `j` |
| `threshold_value` | `float32 (N_node,)` | `j / denominator` |
| `parent_node_id` | `int32 (N_node,)` | 同一树中的父节点编号；根节点为 `-1` |
| `children_offsets` | `int64 (N_node+1,)` | 同时切分 `children_node_id` |
| `children_node_id` | `int32 (L_child,)` | 同一树中的子节点编号 |
| `node_voxel_offsets` | `int64 (N_node+1,)` | 同时切分 `node_voxel_global_linear_index` |
| `node_voxel_global_linear_index` | `int64 (L_voxel,)` | 来源完整图 `(D,H,W)` 的全局 C-order 线性体素编号，等价于 `np.ravel_multi_index((z,y,x),(D,H,W))`, 它不是 BOX 内坐标 |
| `voxel_count` | `int32 (N_node,)` | 必须等于相应 `node_voxel_offsets` 段长度 |
| `bbox_min_zyx` | `int32 (N_node,3)` | 包围盒最小体素索引，端点包含 |
| `bbox_max_zyx` | `int32 (N_node,3)` | 包围盒最大体素索引，端点包含 |
| `centroid_zyx` | `float32 (N_node,3)` | 连续体素索引空间中的质心 |
| `probability_mean` | `float32 (N_node,)` | 节点体素的平均概率 |
| `probability_max` | `float32 (N_node,)` | 节点体素的最大概率 |
| `candidate_eligible` | `bool (N_node,)` | `True` 表示节点可作为正式候选；`False` 表示不可 |
| `ineligible_reason_code` | `uint8 (N_node,)` | 候选资格原因码 |
| `gauss_score` | `float32 (N_node,)`，可选 | 独立 Gauss scorer 的节点分数；只对本次指定 Fα-centered 角色的来源节点为有限值，其余节点为 `NaN` |
| `gauss_selected` | `bool (N_node,)`，可选 | 独立 Gauss scorer 的保留决定；没有有限 `gauss_score` 的节点固定为 `False` |

`children_offsets[i:i+2]` 给出节点 `i` 在 `children_node_id` 中的半开区间；首值必须为 0，末值必须等于 `L_child`。`node_voxel_offsets` 同理切分 `node_voxel_global_linear_index`，首值为 0，末值为 `L_voxel`。每个节点的体素编号在自己的段内升序且不重复。

`gauss_score` 与 `gauss_selected` 必须同时存在或同时缺席。它们属于某个已完成 Fα-centered 角色的独立评分结果，不改写 `candidate_eligible`，也不改变 `clg.npz`、CLG-centered 或 Selector 的有效节点和候选集合。CLG 与 Selector 继续按原有字段工作，并忽略这两个可选字段。Gauss scorer 的四个正参数与固定距离截断单独保存在 producer 级 `gauss_scorer/calibration.json`，不重复写入每个 PDB 的 `forest.npz`。

#### 7.1.1 Gauss scorer 数值定义

Gauss scorer 处理命令指定的 Fα-centered 文件中由 `(source_tree_id, source_node_id)` 指向的来源节点。对节点 `j` 的每个 A 原子 `i`，先计算该原子到来源 blob 任一体素中心的最近世界坐标距离 `d_ji`。固定截断距离为 5 Å；超过该距离的原子权重为 0。其余原子的权重为：

`w_ji = exp(-d_ji² / (2 * tau_angstrom²))`

令 `p_i` 为 `A_probability` 中同一原子的结合概率，则：

- `positive_sum_j = sum_i(w_ji * p_i)`；
- `negative_sum_j = sum_i(w_ji * (1 - p_i))`；
- `gauss_score_j = probability_mean_j + lambda_positive * positive_sum_j - lambda_negative * negative_sum_j`；
- `gauss_selected_j = gauss_score_j >= gauss_score_min`。

两个高斯和都是直接求和，不按原子数或权重和归一化。`lambda_positive`、`lambda_negative`、`tau_angstrom` 和 `gauss_score_min` 都必须是有限正数。它们只用 calibration 集合选择一次，validation 与 train 复用同一份冻结参数。

#### 7.1.2 Gauss scorer 的增量发布

正式 CPU 入口是 `训练与运行/sh/infer/Find_0_Gauss.sh`，Python 入口是 `python -m src.inference.Gauss_Scorer.cli`。Fα 模式消费已经完成的 `probability`、`components` 和指定 Fα-centered；Li 模式只消费独立根中已经完成的 `Li_centered`。两者都不加载模型，也不创建新的 PDB 角色完成标记。

回填对每个 PDB 复用根目录 `_RUNNING` 租约。前置角色尚未完成时记录 `pending`，租约正被 GPU 或其他生产者持有时记录 `skipped_running`，随后继续扫描其余 PDB。重复运行同一清单与分片会补齐后来完成的 PDB。正式 CLI 的 `--force-overwrite` 默认开启：已有两个 Gauss 字段时只替换这两个字段，forest 其他字段保持不变；`--no-force-overwrite` 则只接受逐值相同的幂等结果。仅存在一个 Gauss 字段始终视为损坏并拒绝覆盖。

历史 F1 参数文件继续位于 `{output_root}/Find_0/gauss_scorer/calibration.json`；其他 Fα 或 Li 角色位于 `{output_root}/Find_0/gauss_scorer/{centered_role}/calibration.json`。`selected_parameters` 至少包含 `lambda_positive`、`lambda_negative`、`tau_angstrom` 和 `gauss_score_min`，并记录固定的 `distance_cutoff_angstrom`。该文件还保存 calibration 清单、角色和指标身份；这些运行身份不重复写入每个 PDB 的 NPZ。

原因码：

| 值 | 含义 |
| --- | --- |
| `0` | 合格 |
| `1` | 体素数小于 `min_voxels` |
| `2` | 体素数大于 `max_voxels` |
| `3` | 节点包围盒不能被已解析的合法 80³ BOX 完整包含 |

### 7.2 `components/clg.npz`

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `CLG_id` | `int32 (N_CLG,)` | 从 0 开始连续编号 |
| `tree_id` | `int32 (N_CLG,)` | 当前组件谱系组所属组件树 |
| `CLG_seed_node_id` | `int32 (N_CLG,)` | `t_F1` 层合格种子节点在 `tree_id` 指定 forest 树内的（局部）编号 |
| `CLG_oldest_node_id` | `int32 (N_CLG,)` | 用于解析 centered BOX 的最老节点在 `tree_id` 指定 forest 树内的编号 |
| `candidate_offsets` | `int64 (N_CLG+1,)` | 同时切分下面两个候选值表 |
| `candidate_node_id` | `int32 (N_candidate,)` | 当前树内的候选节点编号 |
| `candidate_threshold_grid_index` | `int32 (N_candidate,)` | 与候选节点同序的阈值编号 |

第 `i` 个组件谱系组的候选段是 `[candidate_offsets[i], candidate_offsets[i+1])`。首值必须为 0，末值必须同时等于 `candidate_node_id` 和 `candidate_threshold_grid_index` 的第一维长度。

组件谱系组数量上限为：

`min(300, 3 * max(2, N_F1_eligible))`

### 7.3 `components/overlap.npz`

该文件只保存候选组件与真实 occurrence 的正交集；零交集不写入。

| 字段 | dtype 与形状 | 含义与索引目标 |
| --- | --- | --- |
| `candidate_occurrence_offsets` | `int64 (N_candidate+1,)` | 同时切分 `overlap_occurrence_index` 和 `intersection_voxel_count` |
| `overlap_occurrence_index` | `int32 (N_overlap,)` | 索引 `occurrence_id` 与 `occurrence_voxel_count` 第一维 |
| `intersection_voxel_count` | `int32 (N_overlap,)` | 候选组件与相应 occurrence 的正交集体素数 |
| `occurrence_id` | `int32 (N_occ,)` | 升序 occurrence 身份表 |
| `occurrence_voxel_count` | `int32 (N_occ,)` | 与 `occurrence_id` 同序的真实体素数 |

`candidate_occurrence_offsets` 的候选顺序不是 forest 全节点顺序，而是先按 `clg.npz` 的 `CLG_id` 顺序、再按每个 CLG 的 `candidate_node_id` 段顺序展开。第 `j` 个展开候选的交集段是 `[candidate_occurrence_offsets[j], candidate_occurrence_offsets[j+1])`；该段中的每个 `overlap_occurrence_index[k]` 是同文件 `occurrence_id` 与 `occurrence_voxel_count` 的本地行号，`intersection_voxel_count[k]` 则是这个候选与该真实 occurrence 的交集体素数。首值必须为 0，末值必须同时等于两个交集值表的第一维长度。候选自身的体素数通过同一个 CLG 候选身份回指 `forest.npz` 读取。

### 7.4 `components/summary.json`

森林字段：

- `denominator: int`
- `threshold_grid_indices_descending: list[int]`
- `connectivity: int`，固定为 26
- `min_voxels: int`
- `max_voxels: int`
- `n_trees: int`
- `n_nodes: int`
- `ineligible_reason_code: object`，字符串键 `"0"`、`"1"`、`"2"`、`"3"` 映射到原因名
- `layers: list[object]`

原因名精确映射为 `"0":"eligible"`、`"1":"below_min_voxels"`、`"2":"above_max_voxels"`、`"3":"bbox_not_contained_by_resolved_box"`。

每个 `layers` 项包含 `threshold_grid_index: int`、`n_nodes: int`、`n_eligible: int`，以及原因计数 `n_below_min_voxels: int`、`n_above_max_voxels: int`、`n_bbox_not_contained_by_resolved_box: int`。

组件谱系组字段：

- `max_split_events: int`
- `max_merge_events: int`
- `max_nodes_per_CLG: int`
- `n_f1_eligible_seeds: int`
- `n_CLG_cap: int`
- `n_CLG_completed: int`
- `n_CLG_rejected_by_node_cap: int`
- `mean_candidates_per_completed_CLG: float`
- `CLG_cap_reached: bool`

`max_split_events` 与 `max_merge_events` 分别限制一次 CLG 枚举允许跨过的分支和合并事件数；`max_nodes_per_CLG` 限制一个已完成 CLG 可保存的候选节点数。`n_f1_eligible_seeds` 是冻结 `t_F1` 层的合格种子数；`n_CLG_cap` 是当前 PDB 允许发布的 CLG 数上限；`n_CLG_completed` 是实际完成并写入 `clg.npz` 的数量；`n_CLG_rejected_by_node_cap` 统计因候选节点数超过上限而拒绝的枚举结果；`mean_candidates_per_completed_CLG` 是已完成 CLG 的平均候选数。`CLG_cap_reached` 只有在完成数等于 `n_CLG_cap` 且仍有未消费的活跃种子时为 `true`。

## 8. centered NPZ 的共同契约

本节定义七个 Fα-centered、`Li_centered.npz`、`CLG_centered.npz` 和 `Selected_Refined_Centered.npz` 共同使用的字段、数据类型和 offsets 规则；第 9 节在此基础上定义各角色如何决定权威体素成员。

一个 PDB 的同一 centered 角色只发布一个 NPZ。变长表使用 offsets；第 `i` 个归档项或候选项的值段一律是半开区间 `[offsets[i], offsets[i+1])`。

运行时可以把多个有序 80³ BOX 放入同一次完整 wrapper forward，A800 的正式默认批量大小为 10。稠密 V 输出按 batch 第 0 维拆分，Find A 表按 forward 后的 `atom_counts` 连续段拆分；输入 `atom_feat` 在每个 BOX 内通过 `A_global_index` 对齐到 forward 输出 A 行序，形成 `A_feat_L0`。P 表按 `anchor_batch_index` 归属拆分。运行批量只影响执行吞吐，不进入 NPZ schema，也不得改变 `centered_box_index`、entry 顺序、来源身份或任一 offsets/value 对齐关系。

### 8.1 共同归档项字段

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `centered_box_index` | `int32 (N_entry,)` | 必须严格等于 `0..N_entry-1` |
| `box_start_zyx` | `int32 (N_entry,3)` | 完整图中的离散 BOX 起点 |
| `box_shape_zyx` | `uint8 (N_entry,3)` | 每项固定为 `[80,80,80]` |
| `box_origin_world` | `float32 (N_entry,3)` | BOX 角点世界 XYZ，单位 Å |
| `voxel_size_world` | `float32 (N_entry,3)` | XYZ 体素尺寸，单位 Å/voxel |
| `source_tree_id` | `int32 (N_entry,)` | 来源 `components/forest.npz` 的组件树编号 |
| `source_node_id` | `int32 (N_entry,)` | 来源节点在 `source_tree_id` 指定树内的局部编号；必须用 `(source_tree_id,source_node_id)` 唯一定位 forest 节点 |
| `source_threshold_grid_index` | `int32 (N_entry,)` | 来源 forest 节点所在阈值层的整数网格编号 `j`，不是体素索引 |
| `source_threshold_value` | `float32 (N_entry,)` | 同一来源节点的二值化阈值 `j/denominator`；它标识来源组件所在阈值层，不是该归档项任一体素的预测概率 |

### 8.2 共同体素变长表

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `voxel_offsets` | `int64 (N_entry+1,)` | 同时切分下面三个权威体素值表 |
| `voxel_index_local_zyx` | `int16 (L_voxel,3)` | 当前角色权威成员体素在本归档项 80³ BOX 内的离散 ZYX 坐标，各轴范围为 `0..79`；它不是完整图线性索引，具体成员集合由第 9 节对应角色定义 |
| `centered_probability` | `float32 (L_voxel,)` | 当前 centered BOX 重新执行完整模型前向后，经 sigmoid 和模型专属后处理得到的配体概率，再按 `voxel_index_local_zyx` 逐行取值；它不是原始滑窗完整图的概率 |
| `voxel_final` | `float16 (L_voxel,C_voxel)` | 与体素同序的最终体素特征 |
| `voxel_aux_offsets` | `int64 (N_entry+1,)` | 同时切分下面两个辅助体素值表 |
| `voxel_aux_index_local_zyx` | `int16 (L_aux,3)` | 当前 centered 输入 `hardmask == True` 的位置在本归档项 80³ BOX 内的离散 ZYX 坐标；它不是完整图索引，也不表示第 9 节的权威配体成员集合 |
| `voxel_aux_probability` | `float32 (L_aux,)` | 当前 centered 前向的受体辅助头经 sigmoid 后，按 `voxel_aux_index_local_zyx` 逐行取出的概率 |

`voxel_offsets` 首值必须为 0，末值必须同时等于 `voxel_index_local_zyx`、`centered_probability`、`voxel_final` 的第一维长度。`voxel_aux_offsets` 首值必须为 0，末值必须同时等于两个辅助体素值表的第一维长度。

`C_voxel` 由实际 checkpoint 决定，不得假定为 48。完全没有成功载荷且无法确定特征宽度时，`voxel_final.shape` 必须是 `(0,0)`。`voxel_aux_probability` 不是 Find 完整图融合使用的 hardmask。模型可能还返回蛋白主链、核酸主链和配体反距离辅助 logits，但当前 centered NPZ 不保存这些值。

### 8.3 Find 的 P 点表

`unet_c1` 不得出现本组字段。Find 在本组存在时必须整组出现。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `P_offsets` | `int64 (N_entry+1,)` | 同时切分下面四个 P 值表 |
| `P_coord_local_xyz` | `float32 (L_P,3)` | BOX 局部连续 XYZ |
| `P_probability` | `float32 (L_P,)` | P 点概率 |
| `P_feat_L2` | `float16 (L_P,C_L2)` | `outputs["pseudo_density_feat"]`；密度、伪原子类别与界面归一化共同形成的 P 初始表示，也是点骨干网络接收的 P 输入 |
| `P_feat_L3` | `float16 (L_P,C_L3)` | `outputs["pseudo_feat_before_interaction"]`；点骨干网络处理完成、A↔P 交叉注意力发生之前的 P 最终表示 |

`P_offsets` 首值必须为 0，末值必须等于四个 P 值表的第一维长度。P 表保存当前 BOX 的全部 P 点，不表示某个 CLG 候选的成员集合，因此没有 `candidate_P_index`。

### 8.4 Find 的 A 原子表

`unet_c1` 不得出现本组字段。Find 在本组存在时必须整组出现。

| 字段 | dtype 与形状 | 含义与索引目标 |
| --- | --- | --- |
| `A_offsets` | `int64 (N_entry+1,)` | 同时切分下面八个 A 值表 |
| `A_global_index` | `int64 (L_A,)` | 索引完整 `receptor_tokens.npz` 第一维，仅用于原子身份和来源追踪 |
| `A_coord_local_xyz` | `float32 (L_A,3)` | BOX 局部连续 XYZ |
| `A_coord_centered_world` | `float32 (L_A,3)` | 相对 BOX 中心的世界 XYZ，单位 Å |
| `A_probability` | `float32 (L_A,)` | A 原子概率 |
| `A_feat_L0` | `float32 (L_A,49)` | 当前 centered 输入 `batch["atom_feat"]` 中、按 `A_global_index` 对齐到模型输出 A 行序的原始受体特征；位于点侧嵌入层之前并保留 float32 精度 |
| `A_feat_L1` | `float16 (L_A,C_L1)` | `outputs["A_feat_L1"]`；点侧嵌入与界面归一化完成、真实原子密度调制发生之前的 A 表示 |
| `A_feat_L2` | `float16 (L_A,C_L2)` | `outputs["A_feat_L2"]`；真实原子密度调制完成后送入点骨干网络的 A 输入表示 |
| `A_feat_L3` | `float16 (L_A,C_L3)` | `outputs["real_feat_before_interaction"]`；点骨干网络处理完成、A↔P 交叉注意力发生之前的 A 最终表示 |

`A_offsets` 首值必须为 0，末值必须等于八个 A 值表的第一维长度。这里的“来源组件”就是当前 centered 条目对应的预测 blob：Fα 角色使用对应阈值层节点，Li 使用本地 Li blob，CLG 使用最老来源节点，Selected 使用被选中的来源节点。A 表保存完整受体原子表中同时落入当前 80³ 核心并位于该预测 blob 体素集合 10 Å 包络内的原子。`A_feat_L0` 已直接持久化；`A_global_index` 仍保留原子身份追踪语义。

Stage1-Find 前向计算仍可产生交叉注意力后的 `A_feat_L4` 与 `P_feat_L4`，但 centered 归档不保存这两组张量。Selector 只读取本节列出的 L3 及以前特征；A/P 分类概率仍使用 Stage1-Find 原有分类头结果。

完全为空的 Find F1 或 CLG 归档无法从载荷确定 P/A 特征宽度时，P 组和 A 组可以同时整体缺席。Selected 归档至少有一个成功项产生相应模态时才保存整组字段；未成功项的 P/A offsets 段必须为空。

## 9. centered 角色的权威成员语义

本节只补充第 8 节共同结构无法表达的角色差异：每个归档项的 `voxel_index_local_zyx` 究竟对应哪一个体素集合。各角色都重新执行当前 80³ BOX 的完整模型前向，因此 `centered_probability`、`voxel_final` 和适用的 P/A 表都来自本次 centered 前向；角色差异只在权威体素集合如何确定。

### 9.1 七个 `centered/F_{alpha}_centered.npz`

alpha 等于 1 时由现有 F1 主线生成 `F1_centered.npz`；其余六个角色由 `produce-falpha --alpha <分数>` 单独补充。七个文件字段完全相同，且都只读取现有 forest：

- 每个归档项对应当前 alpha 冻结阈值层中一个 `candidate_eligible == True` 的组件。
- 排序键依次是 `probability_mean` 降序、`tree_id` 升序、`node_id` 升序。
- `voxel_index_local_zyx` 的权威成员集合，是由原始滑窗融合 `probability_map` 在当前 alpha 阈值上构建的来源 forest 组件；完整图成员由 `components/forest.npz` 的 `node_voxel_global_linear_index` 给出，再换算为当前 BOX 内 ZYX 坐标。
- 这些坐标上的 `centered_probability` 和 `voxel_final` 来自当前 centered 重算，不复用滑窗融合概率或滑窗特征。
- 文件只包含第 8 节的共同字段和适用模态字段，没有额外角色字段。
- 没有合格组件时可以发布 `N_entry == 0` 的空归档；所有 offsets 仍保留唯一的 0。

### 9.2 `centered/CLG_centered.npz`

生产命令：`cal-produce-f1-clg`、`val-produce-prob-f1-clg` 或 `train-produce-prob-f1-clg`。

每个组件谱系组使用 `CLG_oldest_node_id` 的来源组件解析 80³ BOX。`voxel_index_local_zyx` 的权威成员集合就是该最老 forest 节点的完整图组件成员换算到当前 BOX 后的坐标；这些坐标上的概率和特征仍来自当前 centered 重算。归档还保存：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `CLG_id` | `int32 (N_entry,)` | 与 `components/clg.npz` 中同名字段同序 |
| `CLG_seed_node_id` | `int32 (N_entry,)` | 种子节点在 `source_tree_id` 指定 forest 树内的局部编号；与 `source_tree_id` 组成完整节点身份 |
| `CLG_oldest_node_id` | `int32 (N_entry,)` | 最老节点在 `source_tree_id` 指定 forest 树内的局部编号；与 `source_tree_id` 组成完整节点身份，并决定本归档项的 BOX 和权威体素集合 |
| `candidate_offsets` | `int64 (N_entry+1,)` | 同时切分候选节点与阈值编号 |
| `candidate_node_id` | `int32 (N_candidate,)` | 候选节点在所属归档项 `source_tree_id` 指定 forest 树内的局部编号 |
| `candidate_threshold_grid_index` | `int32 (N_candidate,)` | 候选 forest 节点所在阈值层的整数网格编号 `j` |
| `candidate_voxel_offsets` | `int64 (N_candidate+1,)` | 切分 `candidate_voxel_index` |
| `candidate_voxel_index` | `int32 (L_candidate_voxel,)` | 当前候选引用所属归档项权威体素值表的局部行号；它既不是 BOX 内 ZYX 坐标，也不是完整图线性索引 |
| `candidate_A_offsets` | `int64 (N_candidate+1,)` | 切分 `candidate_A_index`；只在 Find A 组存在时出现 |
| `candidate_A_index` | `int32 (L_candidate_A,)` | 当前候选在所属归档项 `A_offsets` 段内的局部值表编号 |

`candidate_offsets` 首值为 0，末值等于两个候选值表长度。`candidate_voxel_offsets` 首值为 0，末值等于 `candidate_voxel_index` 长度。Find A 组存在时，`candidate_A_offsets` 首值为 0，末值等于 `candidate_A_index` 长度。

若候选属于归档项 `i`，则 `voxel_offsets[i] + candidate_voxel_index[k]` 才是该候选成员在归档级 `voxel_index_local_zyx` 中的实际行号，随后从该行读取 BOX 内离散 ZYX 坐标。`candidate_A_index` 同理引用所属归档项 `A_offsets` 段内的 A 值表局部行。两者都不是坐标，也不是完整图全局索引。

### 9.3 `centered/Selected_Refined_Centered.npz`

生产命令：`python -m src.inference.cli selected-refined`。

若不传 `--selection-root`，命令读取：

```text
<stage1_outputs>/{producer}/{split}/{pdb_id}/selector/selection.npz
```

若传入 `--selection-root <选择根目录>`，命令读取：

```text
<选择根目录>/{producer}/{split}/{pdb_id}/selection.npz
```

选择项恢复到 forest 来源节点后，命令使用当前 centered 重算概率和该来源节点自己的 `source_threshold_value`，在当前 80³ BOX 中重新阈值化并构建 26 邻域连通组件。有多个局部组件时，只在与原始来源组件相交的组件中保留交并比最大的一个；多个 CLG 选择同一来源节点时只发布一次。因此，成功项的 `voxel_index_local_zyx` 是本次 centered 概率新精修出的成员集合，可能与原始滑窗 forest 来源组件不同。

归档项按 CLG 顺序和各 CLG 内的来源候选顺序处理；同一来源节点重复出现时保留第一次。

专属字段：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `refine_status` | `uint8 (N_entry,)` | 每个归档项的精修状态码 |
| `refine_status_names` | Unicode `(3,)` | 固定为 `["success","empty","no_overlap"]` |

状态码：

| 值 | 名称 | 含义 |
| --- | --- | --- |
| `0` | `success` | 至少一个局部组件与来源组件相交；保存最大交并比组件及模型载荷 |
| `1` | `empty` | 阈值化后没有局部组件 |
| `2` | `no_overlap` | 有局部组件，但都不与来源组件相交 |

只有 `success` 项可以携带体素和 P/A 载荷。部分成功时，所有 `voxel_final` 值使用成功载荷的统一 `C_voxel`；未成功项的变长段为空。完全没有成功项时 `voxel_final.shape == (0,0)`。

模型 forward、字段读取或精修实现异常不编码进 `refine_status`。此类异常会使当前 role 不发布 `_COMPLETE`，修复后由续跑流程重新生产。

### 9.4 `centered/Li_centered.npz`

Li 变体从每张已完成的 `probability_map.npz` 独立计算 Li 最小交叉熵阈值，再向上量化为 `ceil(t_raw * denominator) / denominator`。量化后的单层连通组件使用与主线相同的 26 邻域、`min_voxels=10`、最大体素数和 80³ BOX 约束，但不落盘 forest、CLG、`candidate_eligible` 或 Selector 产物。

Li 产物放在与主线分离的输出根，例如 `/storage/penghongen/AdaLigand_stage1_LI_inference`。`source_tree_id` 固定为 0，`source_node_id` 是当前 PDB 内按概率均值稳定排序后的连续局部编号，只用于 Li 文件内部身份，不引用主线 forest。除第 8 节共同字段外，还保存：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `source_probability_mean` | `float32 (N_entry,)` | 每个 Li blob 在来源完整图上的平均概率 |
| `li_threshold_raw` | `float32 (1,)` | 量化前的逐图 Li 阈值 |
| `li_threshold_grid_index` | `int32 (1,)` | 向上量化后的整数网格编号 |
| `li_threshold_applied` | `float32 (1,)` | 实际应用阈值，必须等于网格编号除以分母 |
| `threshold_denominator` | `int32 (1,)` | 阈值整数网格分母 |
| `gauss_score` | `float32 (N_entry,)`，可选 | Li 二阶段 Gauss scorer 的条目分数 |
| `gauss_selected` | `bool (N_entry,)`，可选 | Li Gauss scorer 的保留决定 |

Li 的两个 Gauss 字段必须同时出现或同时缺席；它们不会进入主线 forest，也不会成为 CLG 或 Selector 的过滤条件。

## 10. Selector 产物

### 10.1 目录边界

Selector 运行目录由训练配置或 `--selector-run-dir` 显式指定：

```text
<selector_run_dir>/
├── input_CLG_list.json
├── calibration.json
└── {split}/{pdb_id}/scores.npz
```

代码不强制 `<selector_run_dir>` 的外层实验命名。`selection.npz` 的位置由 `--output` 决定；若要让 `selected-refined` 使用默认寻址，调用方必须把它发布到第 9.3 节的默认位置。

### 10.2 `input_CLG_list.json`

生产入口：

```text
python -m src.selector.train --config <selector.yaml>
```

Selector 训练启动时扫描一次可消费的 `CLG_centered`，并在运行目录冻结清单。同一运行目录后续只允许严格复用相同清单。

| 字段 | 类型与含义 |
| --- | --- |
| `schema_version` | `int`，当前为 `1` |
| `stage1_model_name` | `str`，模型来源 |
| `split_order` | `list[str]`，数据划分顺序 |
| `split_counts` | 对象，键为数据划分名，值为该数据划分的 CLG 数 |
| `split_pdb_counts` | 对象，键为数据划分名，值为该数据划分的 PDB 数 |
| `pdb_ids_by_split` | 对象，键为数据划分名，值为完整 PDB 身份数组；包含零 CLG PDB |
| `items` | 对象数组，每项精确包含 `split: str`、`pdb_id: str`、`CLG_id: int` |

`items` 的顺序依次由 `split_order`、PDB 身份和来源 `CLG_id` 决定；同一 PDB 的项目必须完整复制来源 CLG 顺序。

### 10.3 `{split}/{pdb_id}/scores.npz`

生产命令：

```text
python -m src.selector.inference scores --checkpoint <Selector检查点> --input-clg-list <input_CLG_list.json> --stage1-outputs-root <stage1_outputs> --upstream-root <数据根目录> --selector-run-dir <selector_run_dir> --split <数据划分> --device <cpu或cuda>
```

| 字段 | dtype 与形状 | 对齐关系 |
| --- | --- | --- |
| `CLG_id` | `int32 (N_CLG,)` | 与来源 `components/clg.npz` 同序同值 |
| `CLG_logit` | `float32 (N_CLG,)` | 有限的 CLG 未归一化门控分数；字段名保留 `logit` |
| `CLG_valid_probability` | `float32 (N_CLG,)` | `sigmoid(CLG_logit)`，有限且位于 `[0,1]` |
| `candidate_offsets` | `int64 (N_CLG+1,)` | 必须逐值复制来源 `clg.npz` 的同名字段，同时切分下面两个候选值表 |
| `predicted_max_iou` | `float32 (N_candidate,)` | 有限且位于 `[0,1]` 的候选最大交并比预测 |
| `selection_logit` | `float32 (N_candidate,)` | 有限的候选未归一化结构选择分数；只用于精确反链选择，不是概率或候选质量 |

`candidate_offsets` 首值必须为 0，末值必须同时等于两个候选值表的第一维长度。零 CLG PDB 也发布字段齐全的空表，此时 `candidate_offsets` 精确为 `[0]`。

“反链”表示同一组件树中任意两个选中节点都不存在祖先与后代关系。`selection_logit` 只为这种结构化选择提供能量，不应解释为概率。

### 10.4 `<selector_run_dir>/calibration.json`

生产命令：

```text
python -m src.selector.inference calibrate --selector-run-dir <selector_run_dir> --stage1-outputs-root <stage1_outputs> --input-clg-list <input_CLG_list.json> --stage1-model-name <模型来源> --split calibration
```

命令汇集冻结 calibration PDB 清单中的有限 `CLG_valid_probability`，按实际出现概率的升序唯一值扫描门控阈值，以全局 `M_instance` 首个最大值对应的阈值作为 `tau_G`。没有任何有限 CLG 概率时命令报错，不发布伪校准。

根字段：

| 字段 | 类型与含义 |
| --- | --- |
| `schema_version` | `int`，当前为 `1` |
| `stage1_model_name` | `str`，模型来源 |
| `split` | `str`，正式值为 `"calibration"` |
| `calibration_fitted` | `bool`，固定为 `true` |
| `lambda_count` | `float`，从 Selector 已解析训练配置读取 |
| `coverage_thresholds` | `list[float]`，固定为 `[0.3,0.5]` |
| `scan_definition` | `str`，固定为 `"ascending_unique_actual_CLG_valid_probability"` |
| `pdb_count` | `int`，纳入校准的 PDB 数；包含零 CLG PDB |
| `tau_G` | `float`，冻结门控阈值 |
| `best_curve_index` | `int`，索引 `curve` 第一维 |
| `metrics` | 对象，最佳阈值的全局实例指标 |
| `macro_diagnostic` | 对象，最佳阈值的逐 PDB 算术均值诊断 |
| `curve` | 对象数组，完整阈值扫描曲线 |

`metrics` 和每个 `curve[i].global` 都包含：

- `n_pred_instances: int`、`n_gt_instances: int`；
- 对 `tag` 为 `0p3`、`0p5` 的双向覆盖匹配和一对一匹配精确率、召回率与 F1；字段名分别使用 `coverage_*` 和 `one_to_one_*`，类型均为 `float`；
- `M_instance: float`，四个 F1 的算术均值。

`macro_diagnostic` 和每个 `curve[i].macro_diagnostic` 只包含 `coverage_f1_0p3`、`coverage_f1_0p5`、`one_to_one_f1_0p3`、`one_to_one_f1_0p5`、`M_instance`，类型均为 `float`。每个 `curve` 项还包含 `tau_G: float`。

### 10.5 `selection.npz`

单个 PDB 的生产命令：

```text
python -m src.selector.inference selection --scores <scores.npz> --forest <forest.npz> --clg <clg.npz> --calibration <calibration.json> --output <selection.npz>
```

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `CLG_id` | `int32 (N_CLG,)` | 必须与来源 `clg.npz` 同序同值 |
| `CLG_gate_pass` | `bool (N_CLG,)` | `True` 表示 `CLG_valid_probability >= tau_G`；`False` 表示未通过门控 |
| `selected_candidate_offsets` | `int64 (N_CLG+1,)` | 切分 `selected_candidate_index` |
| `selected_candidate_index` | `int16 (N_selected,)` | 所属 CLG 候选段内的局部编号，段内严格递增且无重复 |

第 `i` 个 CLG 的选择段是 `[selected_candidate_offsets[i], selected_candidate_offsets[i+1])`。首值必须为 0，末值必须等于 `N_selected`。每个局部编号必须小于来源 `candidate_offsets[i+1] - candidate_offsets[i]`。

`CLG_gate_pass == False` 时相应选择段必须为空；`CLG_gate_pass == True` 时，精确且非空的最大后验选择必须至少产生一个局部候选编号。

零 CLG PDB 的 `CLG_id` 和 `CLG_gate_pass` 形状都是 `(0,)`，`selected_candidate_offsets` 精确为 `[0]`，`selected_candidate_index` 形状为 `(0,)`。

## 11. 跨文件对齐与可消费条件

### 11.1 完整图到组件

- `probability_map.shape` 必须等于 `geometry.json` 的 `full_shape_zyx`。
- `probability_map.npz` 与 `geometry.json` 的 `origin_xyz`、`voxel_size_xyz` 转换为 `float32` 后必须分别逐值一致。
- `forest.npz` 中的全局线性体素编号必须落在 `[0, D*H*W)`。
- `threshold_value` 必须等于 `threshold_grid_index / denominator`。
- forest 的 `(tree_id,node_id)`、父子关系、包围盒、`voxel_count` 和原因码必须相互一致。

### 11.2 组件到 centered

- F1 centered 的来源节点必须是 `t_F1` 层合格节点。
- CLG centered 的 `CLG_id`、树、种子节点、最老节点和候选顺序必须与 `components/clg.npz` 对齐。
- `candidate_voxel_index` 必须落在所属归档项的体素段局部范围内。
- Find A 组存在时，`candidate_A_index` 必须落在所属归档项的 A 原子段局部范围内。
- `unet_c1` 不得包含 P/A 字段。

### 11.3 CLG 到 Selector

- `input_CLG_list.json` 必须覆盖冻结 PDB 清单，零 CLG PDB 也必须保留在 `pdb_ids_by_split`。
- `scores.npz` 的 `CLG_id` 与 `candidate_offsets` 必须逐值复制来源 CLG。
- `selection.npz` 的 `CLG_id` 必须与 scores 和来源 CLG 同序。
- `selected_candidate_index` 是所属 CLG 候选段内的局部编号，不是 forest 节点编号。

### 11.4 Selector 到 Selected

- `selected-refined` 必须读取可消费的 `components` 角色、`selection.npz`、`probability/geometry.json`、完整图输入和模型检查点；当前实现不读取 `probability_map.npz`。
- Selected 来源节点必须能由 `CLG_id + selected_candidate_index` 唯一恢复。
- 重复来源节点只发布一次。
- 只有成功项允许携带模型载荷；未成功项的变长段必须为空。

## 12. 稳定读取与验收入口

读取任一 PDB 产物前，消费者应按以下顺序校验：

1. 路径中的模型来源、数据划分和小写 `pdb_id` 与请求一致。
2. 需要的 `status/{role}/_COMPLETE` 存在，且 JSON 中 `output_role` 与角色名一致。
3. PDB 目录中不存在 `_RUNNING`；若当前消费者未显式允许超限产物，也不存在 `_BLOB_EXCEED`。
4. 使用 `allow_pickle=False` 读取 NPZ。
5. 校验精确字段集合、dtype、维度、固定形状和有限值要求。
6. 对每个 offsets 校验：长度正确、首值为 0、单调不减、末值等于本文点名的全部值表长度。
7. 对每个索引字段校验：值落在本文点名的目标数组或所属变长段范围内。
8. 校验完整图概率、几何、组件森林、组件谱系组、交集、居中归档、Selector 分数和选择表的跨文件顺序与身份。
9. 校验 `unet_c1` 不含 P/A；Find 的 P/A 要么整组出现，要么只在本文允许的空归档条件下整组缺席。
10. 校验 Selected 只有成功项携带载荷，且未知特征宽度的全空 `voxel_final` 使用 `(0,0)`。

程序化入口：

- 路径构造：[`src/artifacts/paths.py`](paths.py)
- PDB 运行状态：[`src/artifacts/states.py`](states.py)
- NPZ 打包和精确 schema 校验：[`src/artifacts/io.py`](io.py)
- 推理命令和依赖顺序：[`src/inference/README.md`](../inference/README.md)

## 13. 契约边界

- checkpoint、优化器状态、训练日志和 Selector 外层实验目录名不属于本文定义的推理产物。
- 滑窗起点、高斯 `weight_sum`、运行时缓存和未发布临时文件不持久化。
- 当前 centered NPZ 不保存蛋白主链、核酸主链或配体反距离辅助 logits。
- 修改字段、dtype、坐标语义、状态机或固定寻址时，必须同步修改写入器、冷读校验器、本文和 AdaLigand 的 BOX 级数据契约。
