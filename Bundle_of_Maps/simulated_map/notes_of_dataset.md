# Bundle_of_Maps/simulated_map 模拟密度图说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是训练/推理模拟图链路中的子说明。

本文记录本项目中两类受体模拟密度图的生成目的、输入来源、输出路径和使用边界。它面向新手和后续 AI agent：如果只是想知道训练/推理时应该读哪个模拟密度图目录，优先看本文；如果要实际重新生成 `.cmd` 和 SLURM 脚本，核对 `Bundle_of_Maps/simulated_map/gen_chimera_cmds.py`。

本文只提供阅读入口和当前理解，不替代脚本、服务器文件和下游读取代码。排查路径、文件缺失或训练/推理模拟图来源时，请以 `gen_chimera_cmds.py`、实际 `.mrc` 产物和消费端代码为准。

## 1. 为什么需要模拟密度图

Stage1 数据不是只使用实验 EMDB 密度图。`processedPDB_EMDB_binder` 会把实验 EMDB map、模拟受体密度图、体素标签和 ligand 距离图绑定到同一网格，再切成 BOX 训练样本。

模拟密度图的作用是提供“受体结构本身在目标 EMDB 网格上的密度背景”。它和实验 EMDB 图对齐，帮助模型区分：

- 哪些体素是受体骨架/原子占位形成的背景密度；
- 哪些区域更可能来自 ligand 或 pocket 附近信号；
- 在训练或推理时，模型看到的模拟通道应来自当前场景可用的结构来源。

因此本目录维护两套模拟密度图：

1. **真实结构去配体后的 receptor 模拟密度图**：用于训练数据构造。
2. **cryoatom 建模结构的 receptor 模拟密度图**：用于推理场景，因为推理时没有真实 PDB receptor，只能使用 cryoatom 根据密度图建模得到的结构。

## 2. 两套模拟密度图的区别

| 类型 | 结构来源 | 是否含配体 | 服务器输入结构路径 | 服务器模拟图输出路径 | 主要用途 |
|---|---|---:|---|---|---|
| 真实 receptor 模拟图 | 真实 PDB/CIF 去除配体后的受体结构 | 不含配体 | `/storage/chenzhaoyang/cryo_em/CIF_3.5_atom/{PDB_ID}.cif` | `/storage/penghongen/simulated_receptor_map/{emdb_id}.mrc` | 构造训练/验证/测试数据中的 `emdb_sim` 通道 |
| cryoatom receptor 模拟图 | cryoatom 从 EMDB 建模得到的结构 | 不含配体 | `/storage/chenzhaoyang/cryo_em/result_split/{pdb_id_lower}/{pdb_id_lower}.cif` | `/storage/penghongen/simulated_cryoatom_map/{emdb_id}.mrc` | 推理时构造与训练一致语义的模拟受体密度通道 |

注意：两套模拟图都应该只表示 receptor 背景结构，不应该包含 ligand。真实 receptor 版本通过上游去配体结构得到；cryoatom 版本的输入结构本身就是 cryoatom 建模产物，不含配体。

## 3. 生成脚本与核心参数

生成入口：

```text
Bundle_of_Maps/simulated_map/gen_chimera_cmds.py
```

脚本读取 EMDB-PDB-resolution CSV，为每个 EMDB 选择 `fitted_pdbs` 中第一个有效 PDB，然后生成 Chimera `.cmd` 文件和 SLURM 提交脚本。

关键配置位于脚本顶部：

| 参数 | 真实 receptor 版本 | cryoatom 版本 | 说明 |
|---|---|---|---|
| `CSV_PATH` | `/storage/penghongen/EMDB_PDB_resolution_3.5.csv` 或本地备份 | 同左 | CSV 至少包含 `emdb_id`、`resolution`、`fitted_pdbs` |
| `RECEPTOR_CIF_DIR` | `/storage/chenzhaoyang/cryo_em/CIF_3.5_atom` | `/storage/chenzhaoyang/cryo_em/result_split` | 受体 CIF 根目录 |
| `NEST_OR_NOT` | `False` | `True` | 是否使用 `{pdb_id_lower}/{pdb_id_lower}.cif` 嵌套路径 |
| `SIMU_OUTPUT_DIR` | `/storage/penghongen/simulated_receptor_map` | `/storage/penghongen/simulated_cryoatom_map` | Chimera 产出的 `.mrc` 保存目录 |
| `SERVER_OUTPUT_DIR` | `/home/penghongen/My_Project/Pocket_Plus/Bundle_of_Maps/simulated_map/output` | `/home/penghongen/My_Project/Pocket_Plus/Bundle_of_Maps/simulated_map/cryatom_output` | 服务器侧 `.cmd`、SLURM 和日志目录 |
| `OUTPUT_DIR` | 脚本同级 `output` | 脚本同级 `cryatom_output` | 本地/当前环境生成的命令输出根目录 |
| `VALIDATE_FILES` | 建议 `True` | 建议 `True` | 生成 `.cmd` 前检查 receptor CIF 和 EMDB map 是否存在 |

`NEST_OR_NOT` 是兼容两套结构来源的关键开关：

- `False`：按真实 receptor 平铺目录拼路径：`{RECEPTOR_CIF_DIR}/{PDB_ID}.cif`。
- `True`：按 cryoatom 嵌套目录拼路径：`{RECEPTOR_CIF_DIR}/{pdb_id_lower}/{pdb_id_lower}.cif`。

## 4. Chimera 命令做了什么

每个样本生成的 Chimera 命令大致为：

```text
open {receptor_cif_path}
open {emdb_map_path}
volume #1 step 1
molmap #0 {resolution} onGrid #1
volume #2 save {simu_map_path}
close all
```

含义：

1. 打开受体 CIF。
2. 打开原始 EMDB map，用它定义目标网格。
3. 将 EMDB map 采样步长设为 1。
4. 用 `molmap` 按该 EMDB 的真实分辨率把受体结构模拟成密度图，并通过 `onGrid #1` 对齐到原始 EMDB map 网格。
5. 保存模拟图为 `.mrc`。
6. 关闭当前样本，继续处理下一条。

常见 warning：

```text
[MMLIB:WARNING] read_struct_conn: struct_conn table not found
```

这通常表示 CIF 中没有 `struct_conn` 表。cryoatom 或建模结构常见这种 warning；只要没有 `No such file`、`open failed`、`No atoms`、`volume #2 does not exist`、`Error` 或 `Traceback`，一般不代表生成失败。

## 5. 真实 receptor 模拟图：训练用

### 5.1 输入数据

- EMDB-PDB-resolution CSV：`/storage/penghongen/EMDB_PDB_resolution_3.5.csv`
- receptor CIF：`/storage/chenzhaoyang/cryo_em/CIF_3.5_atom/{PDB_ID}.cif`
- 原始 EMDB map：`/storage/chenzhaoyang/cryo_em/EMDB_3.5/{emdb_id}.map`

`CIF_3.5_atom` 是真实结构去除 ligand 后得到的 receptor。它适合训练数据构造，因为训练阶段的标签、PDB 原子、实验 EMDB 和模拟 receptor 通道都来自真实结构数据链路。

### 5.2 输出数据

- 模拟密度图：`/storage/penghongen/simulated_receptor_map/{emdb_id}.mrc`
- 命令输出根目录：`Bundle_of_Maps/simulated_map/output/`
- 服务器命令输出根目录：`/home/penghongen/My_Project/Pocket_Plus/Bundle_of_Maps/simulated_map/output`

下游 `processedPDB_EMDB_binder` 的训练数据构造默认读取：

```text
/storage/penghongen/simulated_receptor_map/{emdb_id}.mrc
```

然后把它转换/重采样为 `sim_npz/{pdb_id}.npz`，再切成 `emdb_sim_BOX`。

## 6. cryoatom receptor 模拟图：推理用

### 6.1 输入数据

- EMDB-PDB-resolution CSV：`/storage/penghongen/EMDB_PDB_resolution_3.5.csv`
- cryoatom 建模 CIF：`/storage/chenzhaoyang/cryo_em/result_split/{pdb_id_lower}/{pdb_id_lower}.cif`
- 原始 EMDB map：`/storage/chenzhaoyang/cryo_em/EMDB_3.5/{emdb_id}.map`

示例：CSV 中 PDB 为 `5BK4` 时，脚本会读取：

```text
/storage/chenzhaoyang/cryo_em/result_split/5bk4/5bk4.cif
```

PDB ID 在 manifest 中仍记录为大写，例如 `5BK4`；仅文件路径使用小写以匹配 `result_split` 的目录结构。

### 6.2 输出数据

- 模拟密度图：`/storage/penghongen/simulated_cryoatom_map/{emdb_id}.mrc`
- 命令输出根目录：`Bundle_of_Maps/simulated_map/cryatom_output/`
- 服务器命令输出根目录：`/home/penghongen/My_Project/Pocket_Plus/Bundle_of_Maps/simulated_map/cryatom_output`

这套输出只应存放 cryoatom 建模结构对应的模拟图，不应混入真实 receptor 结构生成的模拟图。

### 6.3 为什么推理用 cryoatom 模拟图

训练时可以使用真实 PDB receptor 去配体结构生成模拟密度图；但真实推理时通常没有可用的真实 receptor PDB。为了让推理输入仍然包含“受体结构背景密度”这一通道，需要先用 cryoatom 从实验密度图建模出 receptor-like 结构，再用同样的 Chimera `molmap onGrid` 方式生成模拟密度图。

因此：

- 训练数据：实验 EMDB + 真实 receptor 模拟图。
- 推理数据：实验 EMDB + cryoatom receptor 模拟图。

两者语义一致，来源不同。

## 7. 文件缺失过滤

`gen_chimera_cmds.py` 支持在生成 `.cmd` 前检查输入文件是否存在：

- receptor CIF 是否存在；
- EMDB map 是否存在。

默认 `VALIDATE_FILES = True`，适合在服务器上运行。缺失文件会被跳过，不写入 `.cmd`，避免 Chimera 在执行时产生大量连锁错误。

在本地 Windows 上运行时，服务器路径不可访问，应显式跳过检查：

```powershell
python Bundle_of_Maps\simulated_map\gen_chimera_cmds.py --no-validate_files
```

服务器上正式生成时建议使用默认检查：

```bash
conda activate Pocket_Plus_centos7_cu121_allgpu
python /home/penghongen/My_Project/Pocket_Plus/Bundle_of_Maps/simulated_map/gen_chimera_cmds.py
```

## 8. 产物目录结构

以 cryoatom 版本为例，`cryatom_output/` 结构为：

```text
cryatom_output/
    cmd/
        batch_000.cmd
        batch_001.cmd
        ...
    slurm/
        run_all.sh
        submit_batch_000.sh
        submit_batch_001.sh
        ...
    log/
        batch_000.log
        batch_000.err
        ...
    manifest.csv
```

`manifest.csv` 记录每条有效任务：

| 字段 | 意义 |
|---|---|
| `emdb_id` | 标准化 EMDB ID，例如 `emd_63092` |
| `pdb_id` | 标准化 PDB ID，例如 `9LHB` |
| `resolution` | 用于 `molmap` 的真实分辨率 |
| `simu_map_path` | 服务器上 `.mrc` 输出路径 |
| `batch` | 所属 `.cmd` 批次 |

## 9. Stage1 读取关系

当前 Stage1 训练不读取预先切分的 `emdb_sim_BOX`。AdaLigand A—G 数据处理管线先把模拟密度图转换成以下整图产物：

```text
${ADALIGAND_DATA_ROOT}/density/{pdb_id}/sim.npz
```

`src/datasets/stage1_dataset.py` 在物化一个冻结 BOX 请求时读取该文件的 `grid`、`voxel_size` 和 `origin`，再按照 `box_start_zyx` 裁剪 80³ 模拟密度。Find 配置随后由 `density_channel_config.enabled_channels` 决定模拟密度参与哪些模型输入通道；`unet_c1` 只使用实验密度。

如果新增推理数据集或切换模拟图来源，需要同时核对 A—G 数据处理管线写入的 `density/{pdb_id}/sim.npz` 与实验密度 `density/{pdb_id}/exp.npz` 是否具有相同形状、体素大小和世界坐标原点，不能把训练用真实受体模拟图和推理用 cryoatom 模拟图混入同一产物根目录。

## 10. 维护检查清单

修改或重跑模拟密度图时，至少检查：

1. `RECEPTOR_CIF_DIR` 是否指向正确结构来源。
2. `NEST_OR_NOT` 是否匹配目录结构。
3. `SIMU_OUTPUT_DIR` 是否与结构来源一致。
4. `SERVER_OUTPUT_DIR` / `OUTPUT_DIR` 是否与本次输出目录一致。
5. `manifest.csv` 中 `pdb_id`、`simu_map_path` 是否符合预期。
6. `.cmd` 中 receptor CIF 路径和 `.mrc` 输出路径是否符合预期。
7. `.err` 中是否只有可接受 warning，而没有 `No such file`、`open failed`、`No atoms`、`volume #2 does not exist`、`Error` 或 `Traceback`。
8. 下游 `processedPDB_EMDB_binder` 配置是否读取了正确的 simulated map 目录。
