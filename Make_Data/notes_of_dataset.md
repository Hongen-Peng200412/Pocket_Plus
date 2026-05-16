# Make_Data 数据集说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是训练数据链路中的子说明。

本文记录 `Make_Data` 当前数据处理逻辑、服务器落盘路径与维护方式。重点记录目前主用的 `/home/penghongen/My_Project/Data/DATA_v2_raw4`；其它同系列数据集只做概览。

本文是数据导航和背景说明，不是字段、shape 或路径的最终证据。修改或排查问题时，请同时核对生成脚本、下游读取代码和真实 `.npz` / `.json` / 日志；如果说明与实现冲突，以实现和真实产物为准。

## 1. 服务器数据存放路径总览

### 1.1 主用数据集：DATA_v2_raw4

`DATA_v2_raw4` 是第一阶段 PDB/mmCIF 结构解析与点云级标签生成结果，当前由 `Make_Data/process_and_label.py` 通过 `Make_Data/sbatch/process_and_label_v2_raw4.sbatch` 生成。

| 路径 | 内容 | 生成入口 |
|---|---|---|
| `/storage/chenzhaoyang/cryo_em/PDB_3.5/` | 原始 PDB/CIF 输入目录，`process_and_label.py --mode full` 从这里读取结构文件。 | 上游已有数据 |
| `/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/` | 每个成功样本一个 `{pdb_id}/` 子目录，内部保存 `candidates.npz`、`atoms.npz`、`residues.npz`、`graph.npz`、`labels.npz`。 | `process_and_label_v2_raw4.sbatch` |
| `/home/penghongen/My_Project/Data/DATA_v2_raw4/error_logs/` | 解析、配体筛选、主链补全、标签生成等失败样本的错误日志。 | `process_and_label.py` 中的 `return_error_info` |
| `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/` | 对 EMDB-PDB 样本层面的划分 JSON：`all.json`、`train.json`、`val.json`、`test.json`、`test_0.json` 到 `test_5.json`、`train_010.json`、`train_025.json`、`train_050.json`。 | `Make_Data/split_data/generate_full_json.py` |
| `/home/penghongen/My_Project/Pocket_Plus/Make_Data/log/v2_raw4/` | 本仓库保留的 raw4 运行日志副本：`process_and_label_out_241002_*.txt` 与 `process_and_label_err_241002_*.txt`。 | 从服务器 sbatch 输出整理而来 |

`process_and_label_v2_raw4.sbatch` 中 raw4 的关键参数为：

- `MODE="full"`：从结构文件重新解析并生成全部特征与标签。
- `INPUT_DIR="/storage/chenzhaoyang/cryo_em/PDB_3.5/"`。
- `OUTPUT_DIR="/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/"`。
- `ERROR_DIR="/home/penghongen/My_Project/Data/DATA_v2_raw4/"`。
- `CLASS_NAME="five_class_raw4"`。
- `--allow_incomplete_backbone`：允许主链缺失时尝试几何补全，补全状态写入 `residues.npz` 的 `backbone_complete`。
- `--parse_atoms --parse_residues --parse_graph`：raw4 会生成原子、残基和图三类特征。
- Slurm array 为 `0-2`，每个 array task 内 `N_JOBS=8`。
- task 0 在全部 array 结束后调用 `compute_statistics.py --input_dir "$OUTPUT_DIR"` 输出全局统计。

### 1.2 同系列的其它 3 套 Make_Data 数据集

服务器上同系列还产出了 3 套与 raw4 类似的数据集：

| 数据集 | 主要输出目录 | 样本划分目录 | 主要差异 |
|---|---|---|---|
| `DATA_v2_mod4` | `/home/penghongen/My_Project/Data/DATA_v2_mod4/parsed_pdb/` | `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod4/` | 使用 `five_class_mod4`，4 Å 阈值，并要求每类候选配体至少接触 2 个受体残基。 |
| `DATA_v2_raw5` | `/home/penghongen/My_Project/Data/DATA_v2_raw5/parsed_pdb/` | `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw5/` | 使用 `five_class_raw5`，5 Å 阈值，不限制多肽/核酸配体聚合长度。 |
| `DATA_v2_mod5` | `/home/penghongen/My_Project/Data/DATA_v2_mod5/parsed_pdb/` | `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod5/` | 使用 `five_class_mod5`，5 Å 阈值，并要求至少接触 2 个受体残基。 |

这三套与 raw4 的目录结构和落盘文件格式一致，主要区别来自 `Make_Data/labels/filter_config.py` 中的筛选预设，以及对应的 sbatch 配置。

### 1.3 原始映射、过滤与测试集来源

`Make_Data/split_data/generate_full_json.py` 生成样本划分时依赖以下服务器文件：

| 路径 | 意义 |
|---|---|
| `/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv` | 原始 EMDB-PDB 映射表，读取 `emdb_id` 与 `fitted_pdbs` 两列；多个 PDB 时取第一个。 |
| `/home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv` | PDB 总过滤表；只有在该 CSV 中出现的 PDB 才能进入 v2 划分。 |
| `/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc` | 实验 EMDB map 目录；划分时要求对应 EMDB 文件存在。 |
| `/storage/chenzhaoyang/cryo_em/CIF_3.5_cc_qscore` | 经过 cc/qscore 过滤后的 CIF/PDB 结构目录；划分时要求对应 PDB 文件存在。 |
| `/storage/penghongen/simulated_receptor_map/` | 模拟密度图目录；划分时要求对应模拟图存在。 |
| `/home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_*.csv` | 三个纯蛋白测试子集候选来源。 |
| `/home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_*.csv` | 三个含核酸测试子集候选来源。 |

raw4 的划分日志记录：`all.json` 共 5390 个样本，`train.json` 4152 个，`val.json` 269 个，`test.json` 并集 969 个；小训练集为 `train_010.json` 415 个、`train_025.json` 1038 个、`train_050.json` 2076 个。

## 2. 每个落盘数据的意义与生成逻辑

### 2.1 `{pdb_id}/candidates.npz`

位置：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/candidates.npz`

意义：保存过滤前的候选配体属性。这里的候选配体来自 PDB/mmCIF 中几乎所有 HETATM 残基，但已永久排除水、`HETATM_EXCLUSION_LIST` 中的溶剂/缓冲/占位组分，以及与主链共价连接的修饰残基。

生成逻辑：

1. `process_and_label.py` 调用 `PDB_processor/parser.py::parse_structure()`。
2. `parse_structure()` 调用 `PDB_processor/ligand_candidates.py::find_all_hetatm_candidates(model)`。
3. 遍历所有链和残基，仅处理 `het_flag.startswith('H_')` 的 HETATM 残基。
4. 排除水分子并统计 `water_count`。
5. 排除非特异性 HETATM、与主链共价连接的修饰残基和没有重原子的残基。
6. 对每个候选配体计算重原子坐标、重心、重原子数、分子量、金属离子标志、标准氨基酸/核苷酸 HETATM 标志、共价连接标志、HETATM 聚合链长度。
7. `save_candidates_npz()` 压缩保存。

主要字段：

| 字段 | 意义 |
|---|---|
| `n_candidates` | 候选配体数量，不含水。 |
| `water_count` | 被排除的水分子数量。 |
| `resnames` | 每个候选的 CCD 残基名。 |
| `chain_ids` | 每个候选所在链 ID。 |
| `res_ids` | 每个候选残基序号。 |
| `insertion_codes` | 插入码。 |
| `n_heavy_atoms` | 配体重原子数量。 |
| `molecular_weight` | 配体所有重原子的质量和。 |
| `is_metal_ion` | 是否为单原子金属离子。 |
| `is_peptide_like` | 是否为标准氨基酸类 HETATM。 |
| `is_nucleotide_like` | 是否为标准核苷酸类 HETATM。 |
| `is_covalent` | 是否与主链共价连接。 |
| `polymer_length` | 所属连续 HETATM 聚合链段长度；非聚合物为 1。 |
| `centers` | 每个候选配体重心坐标，形状 `(N_cand, 3)`。 |
| `candidate_coords_{i}` | 第 `i` 个候选配体的全部重原子坐标，形状 `(M_i, 3)`。 |

注意：`n_contact_receptor_atoms` 和 `n_contact_receptor_residues` 不写入 `candidates.npz`，因为它们依赖当前筛选预设的 `binding_threshold`，会在标签阶段现场计算。

### 2.2 `{pdb_id}/atoms.npz`

位置：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/atoms.npz`

意义：保存受体原子级坐标与特征。受体包括蛋白质和核酸原子，不包含作为候选配体的非受体 HETATM。

生成逻辑：

1. `parse_structure()` 用 Biopython 解析 PDB/mmCIF 第一个 model；默认遇到多 model 会跳过，除非开启 `--select_first_model`。
2. 遍历标准蛋白/核酸残基；与主链共价连接的修饰残基可并入受体。
3. 跳过氢原子和无法识别元素的原子。
4. 记录每个原子的坐标、元素、所属残基索引、链索引、残基名、原子名。
5. `compute_atom_features(parsed_data, compute_density=True)` 计算 49 维原子特征。
6. `save_atoms_npz()` 保存。

主要字段：

| 字段 | 意义 |
|---|---|
| `coords` | 原子笛卡尔坐标，形状 `(N_atoms, 3)`，单位 Å。 |
| `features` | 原子级特征矩阵，形状 `(N_atoms, 49)`。 |
| `elements` | 元素符号。 |
| `res_indices` | 原子所属残基全局索引，对齐 `residues.npz` 的残基顺序。 |
| `chain_indices` | 原子所属链索引。 |
| `res_names` | 原子所属残基三字母名。 |
| `atom_names` | 原子名。 |

### 2.3 `{pdb_id}/residues.npz`

位置：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/residues.npz`

意义：保存受体残基级坐标、残基特征和局部坐标系。

生成逻辑：

1. `parse_structure()` 按链和残基顺序构建残基表。
2. 蛋白残基代表坐标为 `CA`；核酸残基代表坐标为 `C4'`。
3. 蛋白骨架使用 `N/CA/C`，核酸骨架使用 `C4'/C1'/N1或N9`。
4. raw4 开启 `--allow_incomplete_backbone`，缺失骨架原子时会尝试 `coordinate_reconstruction.py` 中的几何补全；补全成功的残基在 `backbone_complete` 中标记为 `False`，完整原始骨架标记为 `True`。
5. `compute_residue_features()` 计算 33 维残基特征。
6. `compute_local_frames()` 计算局部坐标系与有效 mask。
7. `save_residues_npz()` 保存。

主要字段：

| 字段 | 意义 |
|---|---|
| `coords` | 残基代表原子坐标，形状 `(N_res, 3)`。 |
| `features` | 残基级特征矩阵，形状 `(N_res, 33)`。 |
| `names` | 残基名称。 |
| `types` | 残基类型：`protein` 或 `nucleotide`。 |
| `chain_indices` | 残基所属链索引。 |
| `seq_numbers` | PDB 文件中的残基序列号。 |
| `local_frames` | 每个残基的局部坐标系旋转矩阵，形状 `(N_res, 3, 3)`。 |
| `frames_mask` | 局部坐标系是否有效。 |
| `backbone_complete` | 骨架是否原始完整；`False` 表示由缺失补全逻辑得到。 |

### 2.4 `{pdb_id}/graph.npz`

位置：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/graph.npz`

意义：保存基于受体原子构建的空间图结构。raw4 的 sbatch 开启了 `--parse_graph`，因此会生成该文件。

生成逻辑：

1. `process_and_label.py` 在解析原子后调用 `save_graph_npz(parsed_data, graph_cutoff, ...)`。
2. `graph_cutoff` 默认来自 `PDB_processor/config.py::GRAPH_CUTOFF`。
3. 在距离截断内为原子对构建边。
4. 边权重默认为 `1 / (distance + epsilon)`。

主要字段：

| 字段 | 意义 |
|---|---|
| `edge_row` | 源原子索引。 |
| `edge_col` | 目标原子索引。 |
| `edge_dist` | 两个原子的欧氏距离。 |
| `edge_weight` | 边权重。 |
| `num_atoms` | 原子总数。 |
| `num_residues` | 残基总数。 |
| `cutoff` | 建图距离截断。 |

### 2.5 `{pdb_id}/labels.npz`

位置：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/labels.npz`

意义：保存过滤后的配体实例、逐原子口袋类别和结合位点标签，是下游 `processedPDB_EMDB_binder` 生成体素标签与 BOX 的主要监督来源。

raw4 使用 `five_class_raw4` 预设：

| class_id | class_name | 筛选规则 |
|---|---|---|
| 0 | `background` | 背景，非口袋原子。 |
| 1 | `metal_ion` | 金属离子，binding threshold 4 Å，非共价。 |
| 2 | `peptide` | 氨基酸/多肽类 HETATM，binding threshold 4 Å，非共价，`max_peptide_length=None`。 |
| 3 | `nucleic` | 核苷酸/核酸类 HETATM，binding threshold 4 Å，非共价，`max_nucleic_length=None`。 |
| 4 | `small_molecule` | 非金属、非肽、非核酸的小分子，binding threshold 4 Å，非共价。 |

生成逻辑：

1. `process_and_label.py` 先从 `candidates.npz` 或当前解析结果取得候选配体。
2. 取当前筛选预设中所有规则的最大 `binding_threshold`，raw4 为 4 Å。
3. `compute_contact_attributes()` 用 cKDTree 计算每个候选配体在阈值内接触的受体原子数和受体残基数；受体残基需要至少 2 个原子命中才计为接触残基。
4. `filter_and_classify()` 按 `filter_config.py` 中规则顺序筛选并分配类别；规则优先级从前到后。
5. `compute_binding_labels()` 基于筛选后的配体计算逐原子标签：每个原子可位于多个配体阈值内，但 `instance_ids` 和 `pocket_class_ids` 使用最近配体独占分配。
6. `save_labels_npz()` 保存逐原子字段、逐配体字段、类别名映射以及每个配体的非独占结合原子集合。

主要字段：

| 字段 | 意义 |
|---|---|
| `instance_ids` | 每个受体原子的结合位点实例 ID；背景为 `-1`；实例 ID 等于最近配体的 `candidate_id`。 |
| `ligand_ids` | 每个受体原子最近配体的 `candidate_id`。 |
| `distances` | 每个受体原子到最近配体原子的距离。 |
| `binding_mask` | 任一配体阈值内为 `True`；这是多个配体阈值集合的并集，不受独占约束。 |
| `pocket_class_ids` | 每个受体原子的口袋类别 ID；基于最近配体独占分配，背景为 0。 |
| `num_ligands` | 筛选后保留的配体数量。 |
| `pocket_centers` | 每个保留配体对应口袋的几何中心。 |
| `ligand_resnames` | 保留配体的 CCD 残基名。 |
| `ligand_candidate_ids` | 保留配体的原始 `candidate_id`，与其它逐配体字段同序对齐。 |
| `ligand_class_ids` | 每个保留配体的类别 ID。 |
| `ligand_coords_{id}` | candidate_id 为 `{id}` 的配体重原子坐标。 |
| `pocket_atom_indices_{id}` | candidate_id 为 `{id}` 的配体阈值内全部结合原子索引；不受独占约束，同一原子可属于多个配体。 |
| `pocket_class_name_map` | 类别 ID 到类别名的字符串映射，如 `0:background,1:metal_ion,...`。 |

### 2.6 `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/*.json`

意义：保存样本层面的 EMDB-PDB 划分，用于后续体素化绑定和 BOX 切分。每个 JSON 是 `list[dict[str, str]]`，条目形如 `{"emd_48166": "9MD3"}`。

生成逻辑：

1. 读取 `/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv` 的 EMDB-PDB 映射；EMDB ID 统一为 `emd_数字`，PDB ID 统一大写。
2. 多个 PDB 时取第一个。
3. 用 `/home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv` 过滤 PDB ID。
4. 要求以下文件/目录同时存在：实验 EMDB map、cc/qscore 后的 PDB/CIF、`DATA_v2_raw4/parsed_pdb/{pdb_id}`、模拟密度图。
5. 交集写入 `all.json`。
6. 读取 6 个 BioLiP 测试 CSV，分别生成 `test_0.json` 到 `test_5.json`；它们的并集写入 `test.json`。
7. 从 `all - test` 中随机抽取 5% 全样本量作为 `val.json`。
8. 剩余样本写入 `train.json`。
9. 从 `train.json` 中随机抽取 10%、25%、50% 生成 `train_010.json`、`train_025.json`、`train_050.json`。

raw4 实际数量：

| 文件 | 样本数 |
|---|---:|
| `all.json` | 5390 |
| `train.json` | 4152 |
| `val.json` | 269 |
| `test.json` | 969 |
| `test_0.json` | 269 |
| `test_1.json` | 269 |
| `test_2.json` | 269 |
| `test_3.json` | 54 |
| `test_4.json` | 54 |
| `test_5.json` | 54 |
| `train_010.json` | 415 |
| `train_025.json` | 1038 |
| `train_050.json` | 2076 |

### 2.7 统计输出

`process_and_label_v2_raw4.sbatch` 的 task 0 在 array 任务结束后运行：

```bash
python /home/penghongen/My_Project/Pocket_Plus/Make_Data/compute_statistics.py --input_dir "$OUTPUT_DIR" --n_jobs "$N_JOBS"
```

该脚本不生成固定数据文件，主要向 stdout 打印统计摘要：

- 每个样本过滤前 `candidates.npz` 中候选配体总数和类型标志数量分布。
- 过滤后 `labels.npz` 中各类口袋数量分布。
- 每个独立口袋实例包含的受体结合原子数分布。
- 各类口袋结合原子占受体总原子的比例。

## 3. 维护说明

### 3.1 生成新 Make_Data 数据集时

1. 在 `Make_Data/labels/filter_config.py` 中新增或确认筛选预设，命名为 `YOUR_NAME_PRESET`；CLI/sbatch 中使用的小写名为 `your_name`。
2. 复制并修改 `Make_Data/sbatch/process_and_label_v2_raw4.sbatch`：
   - 修改 Slurm job 名称和日志路径。
   - 修改 `OUTPUT_DIR` 与 `ERROR_DIR` 到新的数据集根目录。
   - 修改 `CLASS_NAME` 到新的筛选预设。
   - 明确是否开启 `--allow_incomplete_backbone`、`--parse_graph`、`--overwrite`。
3. 运行 `process_and_label.py --mode full` 生成 `{pdb_id}/candidates.npz`、`atoms.npz`、`residues.npz`、`graph.npz`、`labels.npz`。
4. 修改 `Make_Data/split_data/generate_full_json.py` 顶部配置中的 `PARSED_PDB_ROOT` 和 `OUTPUT_DIR`，运行生成样本划分 JSON。
5. 将对应 sbatch、日志目录和本文件中的路径/数量同步更新。

### 3.2 更改数据产生逻辑时

如果更改的是候选配体属性、受体解析、原子/残基特征、标签字段或筛选预设，必须同时维护：

- `Make_Data/readme.md`：记录整体流程变化。
- `Make_Data/notes_of_dataset.md`：记录实际服务器路径、落盘字段含义、生成逻辑、数据集版本差异。
- `Make_Data/process_and_label.py` 文件末尾的输出结构说明，如果字段增删或形状变化。
- `Make_Data/split_data/generate_full_json.py`，如果样本筛选或划分逻辑发生变化。
- 下游 `processedPDB_EMDB_binder/notes_of_dataset.md`，如果 `labels.npz` 字段变化会影响体素化绑定或 BOX 切分。

### 3.3 建议的版本命名方式

建议继续使用 `v2_raw4`、`v2_mod4`、`v2_raw5`、`v2_mod5` 这类名字表达两个维度：

- `raw` / `mod`：配体筛选是否包含额外接触残基约束。
- `4` / `5`：binding threshold 是 4 Å 还是 5 Å。

如果以后引入新的字段或更改标签语义，建议升到 `v3_*`，不要复用已有目录名。
