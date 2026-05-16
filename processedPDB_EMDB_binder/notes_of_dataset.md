# processedPDB_EMDB_binder 数据集说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是训练 BOX 数据链路中的子说明。

本文记录 `processedPDB_EMDB_binder` 当前数据处理逻辑、服务器落盘路径与维护方式。重点记录与 `Make_Data` 的 `/home/penghongen/My_Project/Data/DATA_v2_raw4` 对应的 1.0 Å 版本：`/storage/penghongen/Pocket_classic/v2_raw4_10A`。

本文是数据导航和字段背景，不是最终契约。涉及 BOX 字段、体素 shape、类别 ID、mask 或 split 时，请核对生成脚本、训练 dataset/collate、真实 `.npz` / `.json` 和运行日志。

## 1. 服务器数据存放路径总览

### 1.1 主用体素/BOX 数据集：v2_raw4_10A

`v2_raw4_10A` 是第二阶段经典体素数据集：先把 `Make_Data` 生成的点云级 PDB 标签绑定到 EMDB 密度图，再围绕每个口袋切出固定大小 BOX。它依赖：

- 第一阶段点云数据：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/`
- 第一阶段样本划分：`/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/`
- 实验 EMDB map：`/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc`
- 模拟密度图：`/storage/penghongen/simulated_receptor_map`

主目录：`/storage/penghongen/Pocket_classic/v2_raw4_10A/`

| 路径 | 内容 | 生成入口 |
|---|---|---|
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_npz/` | 每个 PDB 一个全图体素标签 `.npz`；把 `labels.npz` 中逐原子 `pocket_class_ids` scatter 到对应 EMDB 网格。 | `bind.py` / `bind_v2_raw4_10A.sbatch` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_npz/` | 每个 PDB 一个重采样后的实验 EMDB 密度图 `.npz`。 | `bind.py` / `bind_v2_raw4_10A.sbatch` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/sim_npz/` | 每个 PDB 一个重采样后的模拟密度图 `.npz`。 | `bind.py` / `bind_v2_raw4_10A.sbatch` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_npz/` | 每个 PDB 一个 ligand 最小距离图 `.npz`，每个口袋类别一个通道。 | `bind.py` / `bind_v2_raw4_10A.sbatch` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_BOX/` | 切块后的标签 BOX；按类别子目录保存。 | `split_and_select_box.py` / `split_and_select_box_CPU_v2_raw4_10A.sbatch` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_exp_BOX/` | 切块后的实验 EMDB BOX；文件名与标签 BOX 对齐。 | `split_and_select_box.py` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_sim_BOX/` | 切块后的模拟 EMDB BOX；文件名与标签 BOX 对齐。 | `split_and_select_box.py` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_BOX/` | 切块后的 ligand 距离图 BOX；文件名与标签 BOX 对齐。 | `split_and_select_box.py` |
| `/storage/penghongen/Pocket_classic/v2_raw4_10A/split/` | BOX 级训练/验证/测试划分 JSON，包含 `split_0`、`split_1`、`split_2`、`split_3`、`split_4`、`split_all`。 | `split_data/generate_full_json.py` |
| `/home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/logs_bind/v2_raw4_10A/` | 本仓库保存的 bind 阶段日志副本。 | sbatch 输出整理 |
| `/home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/logs_split/v2_raw4_10A/` | 本仓库保存的 split 阶段日志副本。 | sbatch 输出整理 |

`bind_v2_raw4_10A.sbatch` 关键参数：

- `EMDB_PDB_JSON="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/all.json"`
- `EMDB_FOLDER_PATH="/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc"`
- `SAMPLE_ROOT_PATH="/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/"`
- `SIM_FOLDER_PATH="/storage/penghongen/simulated_receptor_map"`
- `LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_npz/"`
- `EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_npz/"`
- `SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/sim_npz/"`
- `LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_npz/"`
- `TARGET_VOXEL_SIZE=1.0`
- `NUM_POCKET_CLASSES=4`
- Slurm array 为 `0-2`，每个 array task 内 `N_JOBS=8`。

`split_and_select_box_CPU_v2_raw4_10A.sbatch` 关键参数：

- 输入全图目录为上面四个 `*_npz` 目录。
- `SAMPLE_ROOT_PATH="/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb"`
- `OUTPUT_ROOT_FOLDER="/storage/penghongen/Pocket_classic/v2_raw4_10A/"`
- `WINDOW_SIZE=80`
- `STRIDE=20`
- `R_EXPAND=1.0`
- `EDGE_EXPAND=0.5`
- `ALL_POS_RATIO=0.9`
- `CENTER_POS_RATIO=0.85`
- `CUT_LENGTH=16`
- `SAMPLE_BOX_NUM=20`
- `HARDMASK_UPPER_REQUIRED=0.001`
- Slurm array 为 `0-3`，每个 array task 内 `N_JOBS=8`。

### 1.2 同系列的其它 7 套 processedPDB_EMDB_binder 数据集

服务器上同系列还产出了 7 套与 `v2_raw4_10A` 类似的数据集，组合来自 4 套 Make_Data 版本和 2 个目标体素大小：

| 数据集 | 主目录 | 依赖的 Make_Data 版本 | 目标体素大小 |
|---|---|---|---|
| `v2_raw4_10A` | `/storage/penghongen/Pocket_classic/v2_raw4_10A/` | `DATA_v2_raw4` | 1.0 Å |
| `v2_raw4_15A` | `/storage/penghongen/Pocket_classic/v2_raw4_15A/` | `DATA_v2_raw4` | 1.5 Å |
| `v2_mod4_10A` | `/storage/penghongen/Pocket_classic/v2_mod4_10A/` | `DATA_v2_mod4` | 1.0 Å |
| `v2_mod4_15A` | `/storage/penghongen/Pocket_classic/v2_mod4_15A/` | `DATA_v2_mod4` | 1.5 Å |
| `v2_raw5_10A` | `/storage/penghongen/Pocket_classic/v2_raw5_10A/` | `DATA_v2_raw5` | 1.0 Å |
| `v2_raw5_15A` | `/storage/penghongen/Pocket_classic/v2_raw5_15A/` | `DATA_v2_raw5` | 1.5 Å |
| `v2_mod5_10A` | `/storage/penghongen/Pocket_classic/v2_mod5_10A/` | `DATA_v2_mod5` | 1.0 Å |
| `v2_mod5_15A` | `/storage/penghongen/Pocket_classic/v2_mod5_15A/` | `DATA_v2_mod5` | 1.5 Å |

这些数据集目录结构一致；差异来自上游 `Make_Data` 的配体筛选规则和 `bind.py` 的 `target_voxel_size`。

## 2. 每个落盘数据的意义与生成逻辑

### 2.1 全图体素标签：`pdb_label_npz/{pdb_id}.npz`

位置：`/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_npz/{pdb_id}.npz`

意义：把点云级 PDB 口袋类别标签映射到 EMDB 体素网格上，用于体素分割监督。

生成逻辑：

1. `bind.py::process_single_item()` 根据 `all.json` 中的 `{emd_id: pdb_id}` 找到：
   - EMDB map：`/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc/{emd_id}.map`
   - PDB 样本目录：`/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/`
2. `bind_AtomsLabel_to_EMDB()` 读取 `{pdb_id}/labels.npz` 的 `pocket_class_ids` 和 `{pdb_id}/atoms.npz` 的 `coords`。
3. 通过 `utils/mrc_tools.py::load_map()` 读取 EMDB map，再用 `make_model_grid(..., target_voxel_size=1.0)` 重采样到 1.0 Å。
4. 建立形状 `(1, D, H, W)` 的 `label_np`，背景为 0。
5. 仅对 `pocket_class_ids > 0` 的原子执行世界坐标到体素索引转换：`floor((coord - origin) / voxel_size)`。
6. 如果多个原子落入同一体素，用 `np.maximum.at` 保留较大的类别 ID。
7. 通过 `atomic_np_savez()` 保存。

保存字段：

| 字段 | 意义 |
|---|---|
| `grid` | 标签体素图，形状 `(1, D, H, W)`，数值为 `0..4`。 |
| `voxel_size` | 重采样后的体素大小，1.0 Å 版本通常为 `(1.0, 1.0, 1.0)`。 |
| `origin` | 世界坐标原点，顺序为 `(x, y, z)`。 |

### 2.2 重采样实验密度图：`emdb_npz/{pdb_id}.npz`

位置：`/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_npz/{pdb_id}.npz`

意义：把原始实验 EMDB `.map` 重采样成与标签图完全对齐的 1.0 Å 体素网格。

生成逻辑：

1. `bind.py::process_single_item()` 调用 `load_map(emdb_path)`。
2. 如果 `target_voxel_size` 非空，调用 `make_model_grid()` 重采样。
3. 通过 `atomic_np_savez()` 保存。

保存字段：

| 字段 | 意义 |
|---|---|
| `grid` | 实验密度图，形状 `(D, H, W)`。 |
| `voxel_size` | 体素大小。 |
| `origin` | 世界坐标原点。 |

注意：下游切块时会临时给该 `grid` 加一个 channel 维度，变成 `(1, D, H, W)`。

### 2.3 重采样模拟密度图：`sim_npz/{pdb_id}.npz`

位置：`/storage/penghongen/Pocket_classic/v2_raw4_10A/sim_npz/{pdb_id}.npz`

意义：保存与实验密度图同空间定义的模拟 receptor density map，用于模型输入或对照。

生成逻辑：

1. `bind.py::process_single_item()` 在 `SIM_FOLDER_PATH` 非空时读取 `/storage/penghongen/simulated_receptor_map/{emd_id}.mrc`。
2. 调用 `load_map()` 和 `make_model_grid(..., target_voxel_size=1.0)`。
3. 保存为 `{pdb_id}.npz`。

保存字段与 `emdb_npz` 相同：`grid`、`voxel_size`、`origin`。

### 2.4 ligand 距离图：`ligand_dist_npz/{pdb_id}.npz`

位置：`/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_npz/{pdb_id}.npz`

意义：为每个口袋类别生成一个距离通道。通道 `i` 表示体素中心到第 `i+1` 类 ligand 原子的最近欧氏距离。

生成逻辑：

1. `bind.py::bind_LigandMinDist_to_EMDB()` 读取 `{pdb_id}/labels.npz`。
2. 使用 `ligand_candidate_ids` 与 `ligand_class_ids` 将 `ligand_coords_{candidate_id}` 按类别分组。
3. 加载并重采样 EMDB map，以获得对齐的 `(D, H, W)`、`voxel_size` 与 `origin`。
4. 构造所有体素中心世界坐标。
5. 对每个类别的所有 ligand 原子坐标建 cKDTree，查询每个体素中心到最近同类 ligand 原子的距离。
6. 若某类别在该样本中没有 ligand，则对应通道保持 `np.inf`。

保存字段：

| 字段 | 意义 |
|---|---|
| `grid` | 距离图，形状 `(num_pocket_classes, D, H, W)`；raw4 为 `(4, D, H, W)`。 |
| `voxel_size` | 体素大小。 |
| `origin` | 世界坐标原点。 |

### 2.5 BOX 数据：`*_BOX/{class_name}/{stem}.npz`

位置示例：

- `/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_BOX/metal_ion/{stem}.npz`
- `/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_exp_BOX/metal_ion/{stem}.npz`
- `/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_sim_BOX/metal_ion/{stem}.npz`
- `/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_BOX/metal_ion/{stem}.npz`

类别子目录包括：`metal_ion`、`peptide`、`nucleic`、`small_molecule`，以及随机背景/结构块 `random_BOX`。

文件命名：

- 口袋中心/邻域 BOX：`{pdb_id}_{instance_id}_{Rxx}_{Ryy}_{Rzz}[_C].npz`
  - `pdb_id`：小写 PDB ID。
  - `instance_id`：上游 `labels.npz` 中的 ligand candidate_id，也是口袋实例 ID。
  - `Rxx/Ryy/Rzz`：当前 BOX 中心相对口袋中心的体素偏移。
  - `_C`：中心 BOX 后缀，表示该 BOX 是以口袋中心裁剪出的中心视野。
- 随机 BOX：`{pdb_id}_-1_{rx}_{ry}_{rz}.npz`
  - `instance_id=-1` 表示非特定口袋。
  - `rx/ry/rz` 是随机窗口起点体素索引。

生成逻辑：

1. `split_and_select_box.py::Split_Datas_into_Box()` 读取 `all.json` 并按 Slurm array 分片。
2. 对输入目录取交集：`pdb_label_npz`、`emdb_npz`、`sim_npz`、`ligand_dist_npz` 与 `DATA_v2_raw4/parsed_pdb` 中必须同时存在同一 PDB ID。
3. `_process_one_sample()` 读取上游 `{pdb_id}/labels.npz` 与 `{pdb_id}/atoms.npz`：
   - `pocket_centers`：每个口袋中心世界坐标。
   - `instance_ids`：逐原子实例 ID。
   - `pocket_class_name_map`：类别名称。
   - `ligand_candidate_ids`、`ligand_class_ids`：candidate_id 到类别 ID。
   - `pocket_atom_indices_{id}`：该配体阈值内全部结合原子索引。
   - `ligand_coords_{id}`：配体原子坐标。
4. 若单样本真实配体/口袋超过 50 个，固定随机种子 42 随机保留 50 个进行切块。
5. 读取全图 `pdb_label_npz`、`emdb_npz`、`sim_npz`、`ligand_dist_npz`，构成待切块的 `grids_to_crop`。
6. `Split_Data_into_Box_PocketCentered()` 对每个口袋：
   - 将口袋中心和口袋结合原子坐标从世界坐标转为体素坐标。
   - 用口袋结合原子坐标计算 envelope cuboid 三轴跨度。
   - 大长方体三轴边长为 `window_size * edge_expand + r_expand * envelope_axis_span`。
   - 在大长方体内以 `stride=20` 滑动 `window_size=80` 的窗口，并强制包含中心窗口。
   - 对当前类别生成二值 label mask，只检查 `label == class_id` 的正类体素。
   - 当前 BOX 的正类总数和 inner 区域正类比例需要相对中心 BOX 满足：`all_pos_ratio=0.9`、`center_pos_ratio=0.85`；inner 区域由 `cut_length=16` 裁掉边缘得到。
   - 合格 BOX 同步保存到四个 `*_BOX` 目录的同类别子目录。
7. 如果 `sample_box_num=20`，额外为每个样本随机尝试最多 `20*10` 个窗口；只有 hardmask 中有原子的体素比例大于 `0.001` 才保存到 `random_BOX`。

保存字段：

| 字段 | 意义 |
|---|---|
| `grid` | 当前 BOX 的数据块。标签为 `(1, 80, 80, 80)`；实验/模拟密度为 `(1, 80, 80, 80)`；ligand 距离图为 `(4, 80, 80, 80)`。 |
| `x_range` | 当前 BOX 在世界坐标中的 x 范围。 |
| `y_range` | 当前 BOX 在世界坐标中的 y 范围。 |
| `z_range` | 当前 BOX 在世界坐标中的 z 范围。 |
| `voxel_size` | BOX 对应体素大小。 |
| `origin` | 当前 BOX 左下角近点的世界坐标原点。 |

### 2.6 BOX 级划分 JSON：`split/{split_mode}/{class_name}/*.json`

位置示例：`/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_0/metal_ion/train.json`

意义：保存 BOX 级样本划分。每个 JSON 是 stem 列表，不带 `.npz` 后缀，例如：

```json
["9e01_3_0_0_0_C", "9abc_7_20_0_-20"]
```

生成逻辑：

1. `split_data/generate_full_json.py` 以 `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/` 中的样本级 JSON 为外部划分约束。
2. 分别扫描四个 BOX 根目录：`emdb_exp_BOX`、`emdb_sim_BOX`、`pdb_label_BOX`、`ligand_dist_BOX`。
3. 对每个类别子目录按 stem 取四方交集；只有四个目录同时存在的 stem 才合法。
4. 再用样本级 `all.json` 过滤，保证 BOX 所属 PDB 在 raw4 样本全集内。
5. 对每个 `split_mode` 生成一套目录：
   - `split_0`：只保留 `_C` 结尾的中心 BOX。
   - `split_1`：中心 BOX + 每个结合位点最多 1 个非中心 BOX。
   - `split_2`：中心 BOX + 每个结合位点最多 2 个非中心 BOX。
   - `split_3`：中心 BOX + 每个结合位点最多 3 个非中心 BOX。
   - `split_4`：中心 BOX + 每个结合位点最多 4 个非中心 BOX。
   - `split_all`：保留全部 BOX。
6. `random_BOX` 不受 split_mode 约束，始终取全部合法随机 BOX。
7. 对每个类别和每个样本级 JSON 文件生成同名 BOX stem JSON：`all.json`、`train.json`、`val.json`、`test.json`、`test_0.json` 到 `test_5.json`、`train_010.json`、`train_025.json`、`train_050.json`。

`v2_raw4_10A` 日志中记录的全体合法 BOX 四方交集数量：

| 类别 | stem 数 |
|---|---:|
| `metal_ion` | 60236 |
| `nucleic` | 16 |
| `peptide` | 1560 |
| `random_BOX` | 103207 |
| `small_molecule` | 205848 |
| 合计 | 370867 |

各 split 模式 `all.json` 总 stem 数：

| split | 规则 | all.json 总 stem 数 |
|---|---|---:|
| `split_0` | 中心 BOX + 全部 random_BOX | 174916 |
| `split_1` | 中心 BOX + 每结合位点最多 1 个非中心 BOX + 全部 random_BOX | 236112 |
| `split_2` | 中心 BOX + 每结合位点最多 2 个非中心 BOX + 全部 random_BOX | 279334 |
| `split_3` | 中心 BOX + 每结合位点最多 3 个非中心 BOX + 全部 random_BOX | 315106 |
| `split_4` | 中心 BOX + 每结合位点最多 4 个非中心 BOX + 全部 random_BOX | 334860 |
| `split_all` | 全部 BOX | 370867 |

## 3. 数据流关系

整体数据流为：

```text
Make_Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/atoms.npz + labels.npz
        +
/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc/{emd_id}.map
        +
/storage/penghongen/simulated_receptor_map/{emd_id}.mrc
        ↓ bind.py
/storage/penghongen/Pocket_classic/v2_raw4_10A/{pdb_label_npz, emdb_npz, sim_npz, ligand_dist_npz}/{pdb_id}.npz
        ↓ split_and_select_box.py
/storage/penghongen/Pocket_classic/v2_raw4_10A/{pdb_label_BOX, emdb_exp_BOX, emdb_sim_BOX, ligand_dist_BOX}/{class_name}/{stem}.npz
        ↓ split_data/generate_full_json.py
/storage/penghongen/Pocket_classic/v2_raw4_10A/split/{split_mode}/{class_name}/*.json
```

`Make_Data` 决定“哪个原子属于哪个口袋类别”；`processedPDB_EMDB_binder` 只负责把这些点云标签投到密度图网格，并裁成训练所需 BOX。

## 4. 维护说明

### 4.1 生成新的 processedPDB_EMDB_binder 数据集时

1. 先确认对应 `Make_Data` 数据集已经生成完毕，包括：
   - `/home/penghongen/My_Project/Data/DATA_xxx/parsed_pdb/`
   - `/home/penghongen/My_Project/Data/split/3.5_cc_qscore_xxx/`
2. 复制并修改 `processedPDB_EMDB_binder/sbatch_bind/bind_v2_raw4_10A.sbatch`：
   - 修改 `EMDB_PDB_JSON`。
   - 修改 `SAMPLE_ROOT_PATH`。
   - 修改四个输出目录：`pdb_label_npz`、`emdb_npz`、`sim_npz`、`ligand_dist_npz`。
   - 修改 `TARGET_VOXEL_SIZE`，例如 1.0 或 1.5。
   - 修改 `NUM_POCKET_CLASSES`，保持与上游 `labels.npz` 中非背景类别数一致。
3. 运行 bind 阶段，生成四个全图 `.npz` 目录。
4. 复制并修改 `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_raw4_10A.sbatch`：
   - 输入目录指向 bind 阶段生成的四个全图目录。
   - `SAMPLE_ROOT_PATH` 指向对应 `Make_Data` 数据集。
   - `OUTPUT_ROOT_FOLDER` 指向当前 processed 数据集根目录。
   - 根据体素大小确认 `WINDOW_SIZE`、`STRIDE`、`CUT_LENGTH` 等裁剪参数。
5. 运行 split 阶段，生成四个 BOX 目录。
6. 修改并运行 `processedPDB_EMDB_binder/split_data/generate_full_json.py` 顶部配置，生成 BOX 级 split JSON。
7. 将新数据集路径、参数和数量同步写入本文件。

### 4.2 更改数据产生逻辑时

如果更改以下内容，需要同步维护相关说明和下游读取逻辑：

- `Make_Data/labels.npz` 字段：尤其是 `pocket_atom_indices_{id}`、`ligand_coords_{id}`、`ligand_class_ids`、`pocket_class_name_map`。
- 类别数量或类别名：需要同步 `bind.py --num_pocket_classes`，以及 BOX 类别目录和 split JSON 逻辑。
- 体素化坐标变换：需要确认 `origin`、`voxel_size`、`grid` 轴顺序仍为 `(C, Z, Y, X)`。
- BOX 命名规则：需要同步 `split_data/generate_full_json.py` 的 `get_pdb_id()`、`get_binding_site()` 和 split 规则。
- 新增或删除输入模态：需要同步 `split_and_select_box.py` 的 `name_list_build`、四方交集逻辑和 split JSON 交集逻辑。

### 4.3 建议的版本命名方式

建议继续将 processed 数据集命名为 `{Make_Data版本}_{体素大小}`：

- 上游为 `DATA_v2_raw4`，1.0 Å 体素：`v2_raw4_10A`
- 上游为 `DATA_v2_raw4`，1.5 Å 体素：`v2_raw4_15A`

如果更改 BOX 裁剪规则、随机 BOX 规则或新增输入模态，即使上游 `Make_Data` 不变，也建议使用新的 processed 版本名，避免不同语义的数据混在同一目录。
