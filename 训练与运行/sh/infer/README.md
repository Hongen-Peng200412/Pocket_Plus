# Find_0 正式推理入口

本目录保存 Find_0 的正式推理脚本。脚本不依赖隔离 smoke 的代码或产物；它们从训练运行目录读取已经冻结的 checkpoint 与配置，并把全部正式产物写入：

`/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/`

## 固定模型

- checkpoint：`/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-Find_0-CPC1/Find_0-CPC1____job321743_Find_0_CPC1_lr5e5_p2_val30_chunk2x_2gpu_m8_w1/checkpoints/TOP_epoch_00_score_0.2843.ckpt`
- 训练配置：同一运行目录下的 `config.yaml`
- 选择依据：该 checkpoint 对应的 `val_score/global/voxel_ligand_PRAUC` 为 `0.6754766702651978`，是现存 Find_0 checkpoint 中的最高值。

checkpoint 的 `src_snapshot/src` 提供模型与 wrapper 定义；当前 release 提供推理编排、Dataset 和产物读写代码。隔离 smoke 已用真实 checkpoint、真实配置和 A–G 数据验证这套组合。当前推理代码只做了三个兼容性补齐：模型加载前不提前导入 Dataset；组件生成前先激活 checkpoint 代码快照；无监督推理批次为旧 Find 伪原子注入补充不参与得分的布尔型全零 `atom_label`。

## 正式执行顺序

1. `prepare_inference_pdb_lists.sh`：从已经过滤的 Stage1 集合文件提取、排序并写出三份公共 PDB 清单，同时记录本次 Find_0 推理的路径和启用配置。
2. `Find_0_calibration_probability.sh`：按分片产生 calibration 完整图概率。
3. `Find_0_freeze_thresholds.sh`：在完整 calibration 集合上冻结阈值，并写出语义与实例评估结果；`min_voxels=10`、`max_voxels=2046`、阈值网格分母为 `32768`。后续 calibration、validation 和 train 的组件与 F1-centered 生产都从冻结的 `thresholds.json` 读取同一个体素数下限。
4. `Find_0_calibration_F1.sh`：只补充 calibration 的组件和 `F1_centered`。
5. `Find_0_validation_F1.sh` 与 `Find_0_train_F1.sh`：连续产生完整图概率、组件和 `F1_centered`，不计算 CLG。
6. `Find_0_Gauss.sh`：使用 calibration 集合冻结的参数，为已经完成且静止的 forest 增量回填 `gauss_score` 与 `gauss_selected`；它不重新运行模型，也不影响 CLG 或 Selector 候选。
7. `Find_0_Falpha.sh`：在现有 forest 上按需补充一个 `F_{alpha}_centered.npz`，不重建 forest 或 CLG。
8. `Find_0_Li.sh`：从现有完整图概率生成独立根目录中的 `Li_centered.npz`，不生成 forest、CLG 或 Selector 输入。

以后需要 CLG 时，在相同正式产物根目录运行三个 `*_CLG.sh`。这些脚本使用原有 `*-f1-clg` 命令；已经完成的 probability、components 和 `F1_centered` 会按完成标记跳过，只增加 `CLG_centered`。重复运行同一分片不会重写已经完成的文件。

## Gauss CPU 增量回填

`Find_0_Gauss.sh` 读取同一正式根目录中的冻结参数和公共 PDB 清单。脚本顶部的 `target_split` 选择 calibration、validation 或 train，`global_shard_count` 声明全局分片数，Slurm 数组编号作为 `shard_index`。train 可以采用较大的固定分片数，并只提交当前可用 CPU 所能承担的部分编号。

该任务可以与 validation 或 train 的 GPU 主线同时运行。对每个 PDB，它只在 `probability`、`components` 和 `F1_centered` 都已完成且能够取得根目录 `_RUNNING` 租约时回填；尚未完成的 PDB 记为 `pending`，正在被 GPU 持有的 PDB 记为 `skipped_running`，随后继续处理其他 PDB。GPU 主线结束后再次提交相同分片即可补齐。最终验收要求所有分片汇总后 `n_pending=0`、`n_skipped_running=0`。

正式参数固定读取：

`/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0/gauss_scorer/calibration.json`

一次普通 CPU16 提交示例：

```bash
bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
  --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_Gauss.sh \
  --resource cpu --cpus 16 \
  --job-name find0_gauss
```

`forest.npz` 只增加两个字段。正式脚本显式使用 `--force-overwrite`，因此新冻结参数会替换旧的 `gauss_score` 与 `gauss_selected`，其他字段不变；调试时可改用 `--no-force-overwrite` 要求旧结果逐值一致。仅存在一个字段始终视为损坏并失败。

`Find_0_Gauss.sh` 顶部的 `centered_role` 决定本次回填目标。历史 `F1_centered` 继续读取 `gauss_scorer/calibration.json`；其他 Fα 或 Li 角色读取 `gauss_scorer/{centered_role}/calibration.json`。Fα 写回主线 forest，Li 写回独立的 `Li_centered.npz`；同一主线 forest 只有一对 Gauss 字段，因此最后一次正式 Fα 回填代表当前选定策略。

Gauss 参数采用两阶段搜索。第一阶段保留粗网格和无过滤基线；第二阶段固定第一阶段的 `tau_angstrom` 与 5 Å 截断，围绕最优 `lambda_positive`、`lambda_negative` 各取中心值的 0.8、0.9、1.0、1.1、1.2 倍，围绕 `gauss_score_min` 取 0.3 至 1.7 倍共 15 个值，形成 375 组正参数配置。`ops/Gauss_Scorer/build_refinement_grid_find0.sh` 只生成临时精修网格；数组评估和合并结果仍写入 `/storage/penghongen/tmp`，最终只把第二阶段最优参数冻结到正式 calibration JSON。

## Fα 与 Li 入口

`Find_0_Falpha.sh` 顶部的 `target_split`、`alpha` 和 `global_shard_count` 是需要人工核对的三个主要变量。alpha 等于 `1/1` 时对应已有 `F1_centered`；其余六个值只增加同字段的新 centered 文件，不改变 forest、CLG 或 Stage2/3 既有读取路径。

`Find_0_Li.sh` 读取 `/storage/penghongen/AdaLigand_stage1_inference` 中已完成的概率图，把结果写入 `/storage/penghongen/AdaLigand_stage1_LI_inference`。它固定 `min_voxels=10`，并把逐图 Li 阈值向上量化到 32768 分母网格。Li 结果可独立执行两阶段 Gauss 调参与回填，但不会进入 Selector。

新脚本只在 `target_split=calibration` 时开启 `continue_on_blob_exceed`，因此超限 PDB 保留标记并继续产出；validation 和 train 保持历史停止行为。阈值冻结与 Gauss 正式评估总是开启 `evaluate_on_blob_exceed`，只要所需产物完整就纳入指标。两个开关互不替代。

## 公共 PDB 清单与运行记录

三个数据集合的 PDB 清单由所有 Stage1 模型共同使用，直接保存在推理根目录：

- `/storage/penghongen/AdaLigand_stage1_inference/calibration_pdb_ids.json`
- `/storage/penghongen/AdaLigand_stage1_inference/validation_pdb_ids.json`
- `/storage/penghongen/AdaLigand_stage1_inference/train_pdb_ids.json`

原始集合文件可能为同一 PDB 保存多个 BOX 请求，因此准备脚本只提取 `pdb_id`，转为小写、去重并按名称排序。数据集合划分不变时无需为每个模型重新生成；手动重跑准备脚本会直接更新三份公共清单。

本次 Find_0 的人工可读记录位于 `/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/manifest.json`。它只记录 checkpoint、配置、数据根目录、输出目录、公共清单、模型选择指标和正式启用参数；推理命令不读取该文件，也不把它作为运行条件。

## 分片与资源

三个 calibration 脚本固定使用两个全局分片，提交数组为 `0-1`。Python 按 PDB 清单位置取模，因此两个数组元素各处理 50 个 PDB，合起来完整覆盖 calibration 集合。probability、F1 和 CLG 必须保持相同的两分片总数。

validation 和 train 脚本仍固定使用 16 个全局分片，完整提交数组为 `0-15`。不同 GPU 资源可以共同写入同一正式目录，但分片编号不得重叠；只提交部分编号时，未提交的编号不会被其他数组元素自动处理。

隔离 smoke 使用完整图批量 8，并完成 calibration、validation 和 train 的完整图推理。centered 批量 12 虽然成功，但最高观察到约 81.2/81.9 GiB，因此正式脚本保守使用 centered 批量 8。脚本名称和默认批量不绑定某一种 GPU；正式提交前应根据当时 GPU 的可用显存调整提交资源和批量。

每个进程的 Dataset 缓存上限为 100 GiB。该上限是已读取 PDB 资产的缓存容量，不是 Slurm 内存申请，也不会在进程启动时一次性分配 100 GiB。

正式任务由 `训练与运行/submit_task.sh` 提交。该入口接受本目录脚本的绝对路径或项目内相对路径，并在 allocation 中建立 release、launch 记录和按需启用的锁控制。提交命令必须在人工核对脚本与当前可用显卡后单独执行；本目录脚本本身不会申请资源。

每个 `.sh` 文件顶部都保存了可直接执行的服务器提交命令。以下两条用于开始公共清单准备和 calibration 概率图：

```bash
# 先生成三份公共 PDB 清单，并写下本次 Find_0 的路径与配置记录。
bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
  --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/prepare_inference_pdb_lists.sh \
  --resource cpu --cpus 16 \
  --after_hold \
  --job-name find0_infer_inputs

# 用两个数组元素各申请一张 A800，完成两个 calibration probability 分片。
# cpu96 覆盖 a800 默认使用的 nvlinkg8 QOS；分区仍由 a800 映射为 nvlink。
bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
  --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_calibration_probability.sh \
  --resource a800 --qos cpu96 --gpus 1 --cpus 8 \
  --array '0-1' \
  --after_hold \
  --job-name find0_cal_prob
```

`--after_hold` 表示每个数组元素结束后创建 `try_lock` 并保留资源；省略它时任务结束后自动释放。冻结阈值、calibration F1、validation F1 和 train F1 应分别提交；不要把后一个阶段设置为在前一个阶段尚未完成时启动。
