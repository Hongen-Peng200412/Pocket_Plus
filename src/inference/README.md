# Stage1 推理入口与生产阶段

本文说明 `python -m src.inference.cli` 的九个稳定子命令、每个子命令需要的输入、发布的文件角色、阶段依赖、分片规则和续跑状态。读者无需阅读推理源码即可判断应该执行哪个子命令，以及一个 PDB 的输出能否被后续阶段读取。

命令参数 `--producer` 表示 Stage1 模型来源，当前只允许 `Find_0`、`Find_1`、`Find_2` 和 `unet_c1`。命令参数 `--split` 表示数据划分，正式推理使用 `calibration`、`validation` 或 `train`。`pdb_id` 表示小写 PDB 身份。

完整文件字段、数据类型、数组形状、offsets 切片和空值规则由 [`../artifacts/readme.md`](../artifacts/readme.md) 定义。本文仍列出每个阶段实际生成的文件及其核心内容，使推理入口本身可以独立理解。

## 1. 输入与输出根目录

### 1.1 PDB 清单

`--pdb-list` 接受以下三种文件：

| 文件格式 | 顶层或逐记录内容 |
| --- | --- |
| `.json` | 字符串列表、对象列表，或含 `pdb_ids` 列表的 JSON 对象 |
| `.jsonl` | 每行一个字符串，或每行一个含 `pdb_id` 的 JSON 对象 |
| 其他后缀的文本文件 | 每个非空行一个 PDB 身份 |

所有 PDB 身份会去除首尾空白并转为小写。空清单、空身份和重复身份会使命令失败。清单顺序决定分片归属和处理顺序。

### 1.2 A—G 数据根目录

`--data-root` 指向 Stage A 至 Stage G 的正式数据根目录。推理按 PDB 读取：

| 相对路径 | 推理用途 |
| --- | --- |
| `density/{pdb_id}/exp.npz` | 实验密度、完整图形状、世界坐标原点和体素尺寸 |
| `density/{pdb_id}/sim.npz` | Find 模型使用的模拟密度 |
| `density/{pdb_id}/ligand_area.npz` | calibration 真值及候选组件与真实配体实例的交集 |
| `parse/{pdb_id}/receptor_tokens.npz` | 受体坐标和 Find 原子输入 |

`occurrence` 是磁盘字段沿用的术语，表示一个真实配体实例。`ligand_area.npz` 中的每个 `mask_{occurrence_id}` 保存该实例占据的完整图 ZYX 体素。

### 1.3 checkpoint 与配置

除 `freeze-thresholds` 外，八个模型推理子命令都要求 `--checkpoint`，即模型检查点路径。运行时优先从模型检查点所属训练目录读取：

- `src_snapshot/src/`：训练时冻结的源码快照。
- 唯一存在的 `config.yaml` 或 `resolved_config.yaml`：训练时解析后的配置。

`--config` 可以显式指定解析后配置。模型检查点缺少 `src_snapshot/src/` 时，只有显式传入 `--allow-current-workspace-code` 才允许使用当前工作区源码；该开关不表示当前源码一定与模型检查点相容。

### 1.4 Stage1 输出根目录

`--output-root` 指向 `stage1_outputs` 根目录。一个 PDB 的正式身份为：

```text
{output_root}/{producer}/{split}/{pdb_id}/
```

模型来源级校准文件不按 PDB 分开，位于：

```text
{output_root}/{producer}/calibration/
```

## 2. 九个稳定子命令

### 2.1 子命令与产物

“产物角色”表示可以独立发布完成标记的一组 PDB 文件。五个 PDB 级角色依次为 `probability`、`components`、`F1_centered`、`CLG_centered` 和 `Selected_Refined_Centered`。

| 子命令 | 数据划分 | 必须已经存在的输入 | 本次补齐的产物 |
| --- | --- | --- | --- |
| `cal-probability` | 固定为 `calibration` | 模型检查点、A—G 数据 | `probability` |
| `freeze-thresholds` | 固定为 `calibration` | 清单中全部 PDB 的可读 `probability`、真实配体实例 | 模型来源级 `thresholds.json`、`threshold_scan.npz`、`metrics.json` 和 `_COMPLETE` |
| `cal-produce-f1` | 固定为 `calibration` | 已有 `probability`、模型来源级校准、模型检查点 | `components`、`F1_centered` |
| `cal-produce-f1-clg` | 固定为 `calibration` | 已有 `probability`、模型来源级校准、模型检查点 | `components`、`F1_centered`、`CLG_centered` |
| `val-produce-prob-f1` | 固定为 `validation` | 模型来源级校准、模型检查点、A—G 数据 | `probability`、`components`、`F1_centered` |
| `val-produce-prob-f1-clg` | 固定为 `validation` | 模型来源级校准、模型检查点、A—G 数据 | `probability`、`components`、`F1_centered`、`CLG_centered` |
| `train-produce-prob-f1` | 固定为 `train` | 模型来源级校准、模型检查点、A—G 数据 | `probability`、`components`、`F1_centered` |
| `train-produce-prob-f1-clg` | 固定为 `train` | 模型来源级校准、模型检查点、A—G 数据 | `probability`、`components`、`F1_centered`、`CLG_centered` |
| `selected-refined` | 由 `--split` 指定 | 可读 `components`、`selection.npz`、`geometry.json`、模型检查点、A—G 数据 | `Selected_Refined_Centered` |

### 2.2 最小命令形式

下列占位符含义固定：

- `<PRODUCER>`：`Find_0`、`Find_1`、`Find_2` 或 `unet_c1`。
- `<PDB_LIST>`：第 1.1 节定义的清单文件。
- `<DATA_ROOT>`：第 1.2 节定义的 A—G 数据根目录。
- `<CHECKPOINT>`：Stage1 checkpoint 文件。
- `<OUTPUT_ROOT>`：`stage1_outputs` 根目录。
- `<DEVICE>`：PyTorch 可识别的设备字符串，例如 `cuda`、`cuda:0` 或 `cpu`。

calibration 完整图概率：

```text
python -m src.inference.cli cal-probability --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

冻结模型来源阈值：

```text
python -m src.inference.cli freeze-thresholds --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT>
```

补齐 calibration 的组件与 F1 居中归档：

```text
python -m src.inference.cli cal-produce-f1 --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

补齐 calibration 的组件、F1 与 CLG 居中归档：

```text
python -m src.inference.cli cal-produce-f1-clg --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

连续生产 validation 到 F1 居中归档：

```text
python -m src.inference.cli val-produce-prob-f1 --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

连续生产 validation 到 CLG 居中归档：

```text
python -m src.inference.cli val-produce-prob-f1-clg --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

连续生产 train 到 F1 居中归档：

```text
python -m src.inference.cli train-produce-prob-f1 --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

连续生产 train 到 CLG 居中归档：

```text
python -m src.inference.cli train-produce-prob-f1-clg --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

重跑 Selector 已选组件：

```text
python -m src.inference.cli selected-refined --split <calibration|validation|train> --producer <PRODUCER> --pdb-list <PDB_LIST> --data-root <DATA_ROOT> --checkpoint <CHECKPOINT> --device <DEVICE> --output-root <OUTPUT_ROOT>
```

### 2.3 可选参数

除 `freeze-thresholds` 外的八个 PDB 级推理子命令都接受：

| 参数 | 类型与默认值 | 具体作用 |
| --- | --- | --- |
| `--shard-index` | 整数，默认 `0` | 当前进程的从 0 开始分片编号 |
| `--shard-count` | 正整数，默认 `1` | 总分片数 |
| `--config` | 路径，默认不指定 | 显式指定解析后配置 |
| `--allow-current-workspace-code` | 开关，默认关闭 | checkpoint 缺少源码快照时允许使用当前工作区源码 |
| `--window-batch-size` | 正整数，默认 `1` | 一次模型调用包含的完整图窗口数 |
| `--centered-batch-size` | 正整数，默认 `12` | 一次 centered 完整 wrapper forward 包含的 80³ BOX 数；尾批可以更短 |
| `--cache-max-bytes` | 整数，默认 `536870912000` | 当前推理进程的 Dataset 缓存上限，单位 byte；等于 500 GiB，只限制最多保留多少缓存，不预先分配内存 |

生成 components 的六个子命令还接受：

| 参数 | 类型与默认值 | 具体作用 |
| --- | --- | --- |
| `--max-split-events` | 非负整数，默认 `1` | 一个 CLG 向高阈值扩展时允许的多分支事件数 |
| `--max-merge-events` | 非负整数，默认 `1` | 一个 CLG 向低阈值扩展时允许的多来源合并事件数 |
| `--max-nodes-per-clg` | 正整数，默认 `32` | 一个 CLG 最多包含的候选组件数 |
| `--f1-eligible-limit` | 正整数，默认 `200` | `t_F1` 层合格组件数的继续生产上限 |

`freeze-thresholds` 还接受 `--min-voxels`、`--max-voxels` 和 `--denominator`，默认值依次为 32、2046 和 32768。

`selected-refined` 还接受 `--selection-root`。未指定时读取：

```text
{output_root}/{producer}/{split}/{pdb_id}/selector/selection.npz
```

指定 `<SELECTION_ROOT>` 后读取：

```text
<SELECTION_ROOT>/{producer}/{split}/{pdb_id}/selection.npz
```

## 3. 阶段依赖

每个模型来源的正式生产顺序为：

```text
cal-probability
    ↓
freeze-thresholds
    ↓
cal-produce-f1 ──→ 同目录按需补充 CLG：cal-produce-f1-clg

freeze-thresholds
    ├──→ val-produce-prob-f1 ──→ 同目录按需补充 CLG：val-produce-prob-f1-clg
    └──→ train-produce-prob-f1 ──→ 同目录按需补充 CLG：train-produce-prob-f1-clg

Selector selection.npz
    ↓
selected-refined
```

具体约束：

1. `freeze-thresholds` 要求清单中每个 calibration PDB 的 `probability` 都可读，不允许用部分清单冻结阈值。
2. `cal-produce-f1` 与 `cal-produce-f1-clg` 都不重新生成 probability；缺少任一请求 PDB 的 probability 完成标记时直接失败。
3. validation 和 train 的 F1-only 命令在同一个 PDB 租约内依次补齐 probability、components 和 F1。对应 `*-f1-clg` 命令再增加 CLG；若前三个角色已经有 `_COMPLETE`，它们保持不变，只生成缺少的 CLG。
4. `selected-refined` 要求 `components` 角色可读，并读取 `selection.npz`、`forest.npz`、`clg.npz` 和 `probability/geometry.json`；它不读取 `probability_map.npz`，也不修改已有组件文件。

## 4. 每个阶段发布的文件

### 4.1 probability

```text
probability/probability_map.npz
probability/geometry.json
status/probability/_COMPLETE
```

`probability_map.npz` 精确保存 `probability_map: float32 (D,H,W)`、`origin_xyz: float32 (3,)` 和 `voxel_size_xyz: float32 (3,)`。概率数组空间轴为 ZYX；两个几何数组按世界 XYZ 排列，使单个 NPZ 足以支持概率分析、体素索引到世界坐标的换算和可视化。`geometry.json` 继续保存完整图形状、同值的世界坐标角点原点与 XYZ 体素尺寸、80³ 窗口、40³ 步长和高斯融合参数；两个文件的同名几何字段必须逐值一致。

完整图使用 80³ 窗口和 40³ 步长，不填充；每个轴会补入最后一个合法起点。`hardmask` 是受体占据位置的布尔掩码；值为 `True` 的位置会把配体概率置零。三个 Find 模型来源的正式概率已经应用该掩码，`unet_c1` 不执行该处理。

### 4.2 模型来源级 calibration

```text
{output_root}/{producer}/calibration/
├── thresholds.json
├── threshold_scan.npz
├── metrics.json
└── _COMPLETE
```

`thresholds.json` 保存七个 F-alpha 阈值、`t_F1`、组件体素上下限和 26 邻域连通规则。`threshold_scan.npz` 保存全部阈值编号上的 F-alpha 曲线及 TP、FP、FN。`metrics.json` 保存同一 calibration 数据上的拟合指标，不表示独立 validation 结果。其中 `semantic_dice_micro_t_F1` 先跨 PDB 汇总 TP、FP、FN 再计算 Dice；`semantic_dice_macro_t_F1` 先逐 PDB 计算 Dice 再等权平均。`semantic_tp_t_F1`、`semantic_fp_t_F1` 和 `semantic_fn_t_F1` 是前一种 micro 计算使用的全 calibration 汇总计数。

### 4.3 components

```text
components/forest.npz
components/clg.npz
components/overlap.npz
components/summary.json
status/components/_COMPLETE
```

“组件”表示某个冻结概率阈值上的一个 26 邻域连通体素集合。`forest.npz` 保存不同阈值组件之间的直接包含父子关系。`CLG` 表示组件谱系组，即从一个 `t_F1` 合格组件出发得到的一组谱系候选；`clg.npz` 保存每个组件谱系组的候选组件。`overlap.npz` 保存候选组件与真实配体实例的正交集体素数。`summary.json` 保存构树参数和计数。

### 4.4 F1 与 CLG 居中归档

```text
centered/F1_centered.npz
centered/CLG_centered.npz
status/F1_centered/_COMPLETE
status/CLG_centered/_COMPLETE
```

`centered` 表示把一个来源组件放入不填充的 80³ BOX 后重新执行模型前向，并把稀疏 `voxel_final` 体素表示以及 Find 的 P 点和 A 原子特征保存为一个 PDB 级聚合 NPZ。归档不保存 Stage1 骨干内部的固定多尺度体素网格，也不保存 A/P 交叉注意力后的 L4 特征。Find A 表直接保存 float32 的 49 维 `A_feat_L0` 及 float16 的 L1–L3 特征；`A_global_index` 继续承担原子身份追踪，不要求 Selector 为恢复 L0 再读一次原始受体表。

同一 PDB/role 的有序 BOX 默认按 12 个一批执行完整 wrapper forward。正式 Stage1Dataset 现场裁剪每个 BOX，训练同源 Collator 堆叠 dense V 输入并拼接变长 A 表；forward 后，V 网格按 batch 第 0 维拆分，A 表按模型输出的 `atom_counts` 连续段拆分，P 表按 `anchor_batch_index` 归属拆分。执行批量不改变 `centered_box_index`、来源身份、归档顺序或 offsets 语义；显存不足时可以用 `--centered-batch-size` 下调。

- `F1_centered.npz`：每个条目对应原始滑窗融合概率在 `t_F1` 上形成的一个合格 forest 组件；成员坐标沿用该来源组件，坐标上的概率和特征来自当前 centered 重算。
- `CLG_centered.npz`：每个条目对应一个 CLG，权威体素成员是最老 forest 节点的来源组件；概率和特征来自当前 centered 重算。归档额外保存候选组件在该条目体素表和 A 原子表中的成员行号。

共同字段 `source_threshold_grid_index=j` 和 `source_threshold_value=j/denominator` 标识来源 forest 节点所在阈值层，不是成员体素的概率。`voxel_index_local_zyx` 是当前 80³ BOX 内的离散 ZYX 坐标；`candidate_voxel_index` 则只是引用所属条目体素值表的局部行号，既不是坐标，也不是完整图索引。完整字段和换算公式见 [`../artifacts/readme.md`](../artifacts/readme.md) 第 8、9 节。

### 4.5 Selected 精修归档

```text
centered/Selected_Refined_Centered.npz
status/Selected_Refined_Centered/_COMPLETE
```

Selector 选择先恢复为 `(tree_id, node_id)` 来源组件。推理重新执行该 80³ BOX 的模型前向，再使用来源组件自己的阈值对当前 centered 概率进行二值化，重建 26 邻域组件，并保留与原始滑窗来源组件相交且交并比最大的组件。因此成功条目的体素集合是新精修结果，不要求等于来源 forest 组件；保存的 `centered_probability` 也来自本次 centered 前向，不是原始滑窗完整图概率。

每个来源组件得到一个 `refine_status`：

| 状态码 | 名称 | 含义 |
| --- | --- | --- |
| `0` | `success` | 找到与来源组件相交的局部组件并发布模型载荷 |
| `1` | `empty` | 局部阈值化后没有组件 |
| `2` | `no_overlap` | 存在局部组件，但都不与来源组件相交 |

只有 `success` 条目保存体素和可用的 P/A 模态；其他状态的变长数据段为空。完全没有成功条目且无法确定体素特征宽度时，`voxel_final` 的形状为 `(0, 0)`。

模型 forward、字段读取或精修实现抛出的异常不是领域状态。异常会终止当前 role，且不会发布该 role 的 `_COMPLETE`；调用方修复原因后按现有续跑机制重新执行，不能把异常编码成一个看似可消费的 Selected entry。

## 5. 续跑、互斥与终态

### 5.1 PDB 租约

运行中的 PDB 通过原子创建以下目录获得互斥租约：

```text
{output_root}/{producer}/{split}/{pdb_id}/_RUNNING/
```

`owner.json` 保存处理进程身份、进程号、主机名和创建时间。代码不会自动删除陈旧租约；发现现存 `_RUNNING` 时返回 `skipped_running`。

### 5.2 角色完成标记

每个角色的完成标记位于：

```text
status/{role}/_COMPLETE
```

载荷文件完成原子写入并通过重读校验后，才发布包含 `output_role` 和 `completed_at_utc` 的完成标记。再次请求已经完成的全部角色时返回 `skipped_complete`。

### 5.3 组件超量终态

若 `t_F1` 层合格组件数 `N_F1_eligible` 大于当前 `--f1-eligible-limit`，运行发布：

```text
_BLOB_EXCEED
```

该 JSON 保存实际 `N_F1_eligible` 和本次使用的 `limit`。probability 可以已经完成，但 components 和后续居中角色不会发布。该状态返回 `blob_exceed`，不能当作普通失败自动重跑。

### 5.4 可读条件

一个 PDB 只有同时满足以下条件才可被后续阶段读取：

1. 不存在 `_RUNNING`。
2. 不存在 `_BLOB_EXCEED`。
3. 消费者要求的每个 `status/{role}/_COMPLETE` 都存在。
4. 对应载荷文件能够按 [`../artifacts/readme.md`](../artifacts/readme.md) 的字段契约读取。

## 6. 分片规则

除 `freeze-thresholds` 外，八个 PDB 级推理子命令接受下列分片参数。

`--shard-count n --shard-index i` 选择清单中满足以下关系的记录：

```text
record_index % n == i
```

其中 `record_index` 从 0 开始。各分片互斥并覆盖完整清单；分片内保持原始相对顺序。增加或减少处理进程数后可以用新的分片参数重新扫描，因为已经完成的角色会跳过。

## 7. 读取前检查

执行后续阶段前至少确认：

1. 命令使用的模型来源、数据划分和 PDB 清单与目标目录一致。
2. 模型来源级 calibration 已完整发布，且 `result_scope` 为 `calibration_fitted`。
3. PDB 没有 `_RUNNING` 或 `_BLOB_EXCEED`。
4. 所需角色完成标记和载荷文件同时存在。
5. `selection.npz` 的 `CLG_id` 与来源 `clg.npz` 完全同序。
6. 空 centered 归档遵守 `(0, 0)` 的未知体素特征宽度规则，不假定固定 48 通道。
