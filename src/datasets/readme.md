# Stage1 正式训练清单与 BOX 池

## 本文覆盖什么

本文说明 AdaLigand Stage1 正式训练使用的 `final_keep_list.jsonl`、四份数据划分文件、单 PDB BOX 文件、BOX 发布清单和冻结验证请求。读者不需要阅读 Python 代码，就可以查到每个文件包含哪些字段，以及字段之间如何对齐。

正式产物由 `ops/materialize_filtered_stage1_preparation.py` 从已有 Stage1 preparation 最小复制而来。该脚本不重新选择 train、validation、calibration 或 held-out PDB，不重新计算 BOX 起点，也不扫描辅助标签文件。

## 固定输入与输出位置

源 Stage1 preparation：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000
```

脚本读取的源文件：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory/final_keep_list.jsonl
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/split/train.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/split/validation.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/split/calibration.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/split/held_out_pool.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/box_pool/manifest.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/box_pool/validation_selection.npz
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/box_pool/config.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/box_pool/train/{pdb_id}.npz
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/box_pool/validation/{pdb_id}.npz
```

正式目标根目录：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation
```

脚本写出的正式文件：

```text
stage1_preparation/
├── final_keep_list.jsonl
├── split/
│   ├── train.json
│   ├── validation.json
│   ├── calibration.json
│   ├── held_out_pool.json
│   └── _COMPLETE
└── box_pool/
    ├── train/{pdb_id}.npz
    ├── validation/{pdb_id}.npz
    ├── manifest.json
    ├── validation_selection.npz
    ├── config.json
    └── _COMPLETE
```

正式目标根目录可以预先存在，因为它同时容纳旧版本子目录。脚本只要求新的 `final_keep_list.jsonl`、`split/` 和 `box_pool/` 尚不存在；任何一个已经存在时都会停止，不覆盖文件。

## 固定排除的 PDB

以下 16 个 PDB 缺少本轮训练要求的完整辅助标签：

```text
1q5c  2w49  5y6p  6k0a
7cbp  7cth  7n6g  7ojf
7pel  7wc2  7z8g  8olc
9hhl  9v7i  9wqp  9yx6
```

编号保存在脚本顶部的 `EXCLUDED_PDB_IDS` 全局常量中。脚本会从源 `final_keep_list.jsonl`、四份数据划分文件、`manifest.json` 和 `validation_selection.npz` 中删除实际出现的编号；某个编号在全部源文件中本来就不存在时，脚本输出提示并继续，不会为了重复排除一个已经不存在的 PDB 而停止。

任务 328567 的实际源产物中，`5y6p`、`7n6g`、`7z8g`、`9hhl` 和 `9v7i` 原本存在并已删除；其余 11 个编号原本就不存在。新目录的全部清单和 BOX 文件已经过独立核验，不含上述 16 个编号。

## 训练直接读取的 BOX 池

### 单 PDB BOX 文件

文件位置：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/train/{pdb_id}.npz
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/validation/{pdb_id}.npz
```

一个 NPZ 文件保存一个 PDB 的全部冻结 BOX 候选。所有起点都使用完整密度图的 ZYX 体素编号，表示一个 `80×80×80` BOX 在完整密度图中的最小角点。

| 字段 | NumPy 数据类型 | 形状 | 含义与对齐关系 |
| --- | --- | --- | --- |
| pdb_id | Unicode 标量 | () | 当前文件对应的四位小写 PDB 编号；必须与文件名及 manifest.json 一致。 |
| occurrence_id | int32 | (N_occ,) | 当前 PDB 中进入 BOX 池的候选配体实例编号。数值对应清单中的 candidate_id。 |
| center_start_zyx | int32 | (N_occ,3) | 每个候选配体中心 BOX 的 ZYX 起点；第一维逐元素对齐 occurrence_id。 |
| bias_start_zyx | int32 | (N_occ,30,3) | 每个候选配体的 30 个偏移 BOX 起点；第一维逐元素对齐 occurrence_id，第二维编号范围为 0..29。 |
| context_start_zyx | int32 | (N_context,3) | 当前 PDB 共享的受体环境 BOX 起点；不与某一个 occurrence_id 绑定，允许 N_context=0。 |

脚本不打开并改写保留的单 PDB NPZ，而是直接复制整个文件。命中固定排除编号的文件不进入正式目录。

### `manifest.json`

文件位置：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/manifest.json
```

该文件是 Dataset 发现单 PDB BOX 文件的唯一清单。训练代码不会扫描 `train/` 或 `validation/` 中未列出的文件。

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| schema_version | int | 固定为 1。 |
| splits | object | 包含 train 与 validation 两个数组。 |
| splits.train | array[object] | 训练 BOX 文件清单，顺序继承源 manifest。 |
| splits.validation | array[object] | 验证 BOX 文件清单，顺序继承源 manifest。 |
| splits.*[].pdb_id | str | 单 PDB BOX 文件对应的四位小写 PDB 编号。 |
| splits.*[].path | str | 相对于 box_pool/ 的文件位置；首级目录必须与所在的 train 或 validation 数组一致。 |

脚本删除 `pdb_id` 命中固定排除集合的文件引用，并复制其余引用指向的 NPZ。

### `validation_selection.npz`

文件位置：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/validation_selection.npz
```

该文件保存冻结验证请求。它不复制 BOX 起点，而是使用 PDB 编号、候选配体实例编号和候选 BOX 编号引用 `box_pool/validation/{pdb_id}.npz`。

| 字段 | NumPy 数据类型 | 形状 | 含义与对齐关系 |
| --- | --- | --- | --- |
| validation_pdb_id | 定长字节或 Unicode | (N_pdb,) | 验证 PDB 表。所有 *_pdb_index 数值都索引此数组的第一维。如 ["1abc", "2def"] |
| center_pdb_index | int32 | (N_center,) | 每个中心验证请求对应的验证 PDB 编号。 |
| center_occurrence_id | int32 | (N_center,) | 每个中心请求对应的 occurrence_id；逐元素对齐 center_pdb_index。 |
| bias_pdb_index | int32 | (N_bias,) | 每个偏移验证请求对应的验证 PDB 编号。 |
| bias_occurrence_id | int32 | (N_bias,) | 每个偏移请求对应的 occurrence_id；逐元素对齐 bias_pdb_index。 |
| bias_candidate_index | int16 | (N_bias,) | 对应 bias_start_zyx 第二维的编号，取值范围为 0..29。 |
| context_pdb_index | int32 | (N_context_request,) | 每个受体环境验证请求对应的验证 PDB 编号。 |
| context_candidate_index | int32 | (N_context_request,) | 对应 context_start_zyx 第一维的编号。 |

脚本先从 `validation_pdb_id` 删除固定排除 PDB，再删除引用这些 PDB 的中心、偏移和受体环境请求。保留请求的相对顺序不变；所有保留的 `*_pdb_index` 会重新编号，使其继续指向缩短后的 `validation_pdb_id`。

### `config.json`

文件位置：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/config.json
```

该文件直接复制源 BOX 池配置，因为本次不改变 BOX 生成规则。

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| box_shape_zyx | array[int]，长度 3 | BOX 的 ZYX 体素数，正式值为 [80,80,80]。 |
| bias_candidates_per_occurrence | int | 每个候选配体冻结的偏移 BOX 数，正式值为 30。 |
| bias_radius_formula | str | 根据配体重原子数计算偏移采样半径的公式。 |
| bias_selected_per_epoch | int | 每个训练周期从 30 个偏移候选中选择的数量，正式值为 5。 |
| context_generator | object | 受体环境 BOX 候选的生成规则。 |
| context_generator.sampling | str | 在各轴合法整数起点范围内均匀抽样。 |
| context_generator.target_count | int | 每个 PDB 期望保存的受体环境候选数。 |
| context_generator.max_attempts | int | 寻找合法受体环境起点的最大尝试次数。 |
| context_generator.min_core_receptor_heavy_atoms | int | BOX 核心区域要求包含的最少受体重原子数。 |
| context_generator.ligand_filter | bool | 是否按配体位置过滤受体环境候选；当前为 false(不要求 context BOX 避开配体，也不要求它必须包含配体) |
| occurrence_cap_per_pdb_per_epoch | int | 每个训练周期最多选择的候选配体实例数，正式值为 50。 |
| entry_ratio | object[str,int] | center、bias、context 三类训练请求的数量关系，正式值为 1:5:3。 |
| train_random_rotation_90_degree | bool | 训练时是否允许对 BOX 做随机 90 度空间旋转。 |
| seed | int | BOX 候选生成的固定随机种子。 |
| seed_rule | str | 从基础随机种子、数据划分名称和 PDB 编号派生单 PDB 随机流的规则。 |
| parallel_publisher | object | 旧 BOX 池并行发布过程的记录；本次筛除脚本原样复制该信息，不据此重新生成 BOX。 |
| parallel_publisher.script | str | 旧 BOX 池并行发布脚本，正式记录为 `tmp/stage1_parallel_box_pool.py`。 |
| parallel_publisher.workers | int | 旧 BOX 池发布时记录的并行进程数，正式值为 32。 |
| parallel_publisher.single_pdb_builder | str | 旧流程为单个 PDB 生成 BOX 池时调用的 Python 函数，正式记录为 `src.datasets.stage1_box_pool.build_pdb_box_pool`。 |

### 完成标记与读取关系

`box_pool/_COMPLETE` 是训练读取 BOX 池的发布门槛。`stage1_requests.py` 会在读取 manifest 前要求它存在。脚本只在 manifest、验证请求、配置和全部单 PDB NPZ 写完后创建这个零字节文件。

训练时的读取关系是：

```text
box_pool/manifest.json
    → train/{pdb_id}.npz
    → 每个训练周期按固定 seed 与 epoch 选择 1:5:3 BOX

box_pool/validation_selection.npz
    → validation_pdb_id
    → validation/{pdb_id}.npz
    → occurrence_id 或候选 BOX 编号
```

训练期间，Dataset 直接读取的是 `box_pool/manifest.json`、`validation_selection.npz` 和它们引用的单 PDB NPZ。下面的 `final_keep_list.jsonl` 与四份数据划分文件保存正式数据版本的候选配体身份、质量字段和划分来源，不在每个训练周期中直接读取。

## 未来可能用到的冻结比例npz

当训练配置中的 `box_sample_fraction` 小于 `1.0` 时，训练代码会从完整训练请求和完整验证请求中分别选择固定比例的请求，并把选择结果保存在以下文件：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/train_selection_{fraction}_seed{seed}.npz
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/validation_selection_{fraction}_seed{seed}.npz
```

例如，`box_sample_fraction=0.1`、`request_seed=42` 时，预期保存路径是：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/train_selection_0.1_seed42.npz
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool/validation_selection_0.1_seed42.npz
```

文件名中的 `{fraction}` 最多保留 12 位有效数字，`{seed}` 是训练配置中的整数 `request_seed`。`box_sample_fraction=1.0` 时不会创建这两种文件：训练请求继续随训练周期重新选择，验证直接使用 `validation_selection.npz` 中的全部固定请求。

训练冻结比例文件以训练周期 0 生成的完整 `1:5:3` 请求为来源；验证冻结比例文件以 `validation_selection.npz` 恢复出的全部固定验证请求为来源。两种文件的请求数都等于 `floor(N_full × box_sample_fraction)`，其中 `N_full` 是对应完整请求序列的长度。选择过程分别保持 `center`、`bias` 和 `context` 在完整请求序列中的数量比例，并由 `request_seed` 固定具体选择结果。文件存在时，后续训练直接核对并复用同一份请求，不会在不同训练周期重新抽取。

| 字段 | NumPy 数据类型 | 形状 | 含义与对齐关系 |
| --- | --- | --- | --- |
| pdb_id | Unicode | (N_req,) | 每个冻结请求对应的四位小写 PDB 编号；与本表其余长度为 N_req 的数组逐元素对齐。 |
| box_start_zyx | int32 | (N_req,3) | 每个请求在完整密度图中的 BOX 起点，最后一维依次为 Z、Y、X；每个起点对应一个 `80×80×80` BOX。 |
| role | Unicode | (N_req,) | 请求类别，取值为 `center`、`bias` 或 `context`。 |
| occurrence_id | int32 | (N_req,) | `center` 和 `bias` 请求引用的候选配体实例编号；不适用时保存 `-1`。训练请求中的 `context` 会保留当前配体实例编号，验证请求中的 `context` 保存 `-1`。 |
| candidate_index | int32 | (N_req,) | `bias` 请求保存 `bias_start_zyx` 第二维的编号，范围为 0..29；`context` 请求保存 `context_start_zyx` 第一维的编号；`center` 请求保存 `-1`。 |
| require_targets | bool | (N_req,) | `True` 表示 Dataset 需要为该请求构造监督字段；当前训练和验证冻结比例文件中的值均为 `True`。 |
| box_sample_fraction | float64 | () | 从完整请求序列保留的比例，必须与当前训练配置一致。 |
| request_seed | int64 | () | 选择固定请求子集时使用的整数随机种子，必须与当前训练配置一致。 |
| selection_epoch | int64 | () | 固定为 `0`；表示冻结比例文件基于训练周期 0 的选择且后续训练周期复用同一份请求。 |
| source_manifest_sha256 | Unicode | () | 生成文件时使用的 `box_pool/manifest.json` 内容摘要；用于拒绝读取由另一份 BOX 池生成的冻结请求。 |
| source_validation_sha256 | Unicode | () | 仅存在于验证冻结比例文件；保存 `box_pool/validation_selection.npz` 的内容摘要，用于确认验证请求来源没有变化。 |
| schema_version | uint16 | () | 冻结比例 NPZ 的字段契约版本，当前固定为 `1`。 |

## 来源与划分清单

### `final_keep_list.jsonl`

文件位置：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/final_keep_list.jsonl
```

文件是 UTF-8 JSON Lines。每个非空文本记录表示一个候选配体实例。任务 328567 写出的 577,483 个实例都包含下列 16 个字段：

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `pdb_id` | `str` | 四位小写 PDB 编号。同一 PDB 可以对应多个候选配体实例。 |
| `candidate_id` | `int` | 该 PDB 内候选配体实例的稳定编号。`pdb_id` 与 `candidate_id` 共同确定一个实例。 |
| `type_tag` | `str` | 配体类别。取值为 `small_molecule`、`sugar`、`peptide_like`、`nucleotide_like`、`ion` 或 `other`。 |
| `map_resolution` | `float` | 当前 PDB 所选实验密度图的分辨率，单位 Å。 |
| `cc_all` | `float` | 在实验密度图非零体素内，实验密度与模拟密度不减去各自均值时的相关系数。 |
| `cc_all_about_mean` | `float` | 在同一实验密度图非零体素内，实验密度与模拟密度分别减去各自均值后的相关系数。 |
| `cc_contour` | `float` | 在实验密度高于推荐等高线的体素内，实验密度与模拟密度不减去各自均值时的相关系数。 |
| `cc_contour_about_mean` | `float` | 在同一推荐等高线区域内，实验密度与模拟密度分别减去各自均值后的相关系数。 |
| `q_score` | `float` | 该配体实例中实际存在的重原子 Q-score 均值。 |
| `q_score_median` | `float` | 该配体实例中实际存在的重原子 Q-score 中位数。 |
| `q_score_min` | `float` | 该配体实例中实际存在的重原子 Q-score 最小值。 |
| `pocket_n_atoms` | `int` | 距离该配体任一实际存在重原子不大于 6 Å 的受体重原子数量。受体原子来自 mmCIF 中 `group_PDB=ATOM` 的首个模型。 |
| `pocket_status` | `str` | `ok` 表示上述 6 Å 范围内存在受体原子；`no_receptor_atoms_within_radius` 表示没有受体原子。 |
| `pocket_q_score` | `float` 或 `null` | 上述 6 Å 受体原子的 Q-score 均值；`pocket_n_atoms=0` 时为 `null`。 |
| `pocket_q_score_median` | `float` 或 `null` | 上述 6 Å 受体原子的 Q-score 中位数；`pocket_n_atoms=0` 时为 `null`。 |
| `pocket_q_score_min` | `float` 或 `null` | 上述 6 Å 受体原子的 Q-score 最小值；`pocket_n_atoms=0` 时为 `null`。 |

四个 `cc_*` 字段和所有非空 Q-score 都是有限数值，数值范围为 `[-1,1]`。任务 328567 的正式文件中四个 `cc_*` 字段均为 `float`。三个 `pocket_q_*` 字段始终同时为数值或同时为 `null`；正式 `final_keep_list.jsonl` 中各有 3,077 个 `null`。

脚本只删除 `pdb_id` 命中 `EXCLUDED_PDB_IDS` 的完整记录。其余 JSON 文本和原有顺序保持不变。

### 四份数据划分文件

文件位置：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/split/train.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/split/validation.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/split/calibration.json
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/split/held_out_pool.json
```

每个文件是 UTF-8 JSON array。数组中的每个元素都使用 `final_keep_list.jsonl` 上表所列的同一组 16 个字段，字段类型和含义也完全相同；它不是只包含 `pdb_id` 与 `candidate_id` 的精简索引。一个 PDB 的全部候选配体实例只属于一份数据划分文件。

任务 328567 产物中三个 `pocket_q_*` 字段的 `null` 数量如下：

| 文件 | 每个 `pocket_q_*` 字段的 `null` 数量 |
| --- | ---: |
| `train.json` | 2,570 |
| `validation.json` | 3 |
| `calibration.json` | 3 |
| `held_out_pool.json` | 501 |

四份文件继承源 preparation 已经冻结的划分。脚本只删除固定排除 PDB，不把剩余 PDB 重新排名，也不把它们移动到另一份划分文件。

`split/_COMPLETE` 是零字节完成标记。只有四份 JSON 均写完后才创建。

## 运行入口与安全边界

在 Pocket_Plus 仓库根目录运行：

```bash
python ops/materialize_filtered_stage1_preparation.py
```

脚本没有命令行参数。源路径、目标路径和排除编号只能在阅读并修改文件顶部常量后改变。

脚本不会覆盖已有的 `final_keep_list.jsonl`、`split/` 或 `box_pool/`。它不使用临时发布目录；如果复制过程中出现异常，目标位置可能保留没有 `_COMPLETE` 的未完成文件。重新运行前应先人工检查这些文件，确认不是需要保留的正式产物，再处理未完成内容。
