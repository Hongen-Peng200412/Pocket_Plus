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
3. `Find_0_freeze_thresholds.sh`：在完整 calibration 集合上冻结阈值，并写出语义与实例评估结果；`min_voxels=15`、`max_voxels=2046`、阈值网格分母为 `32768`。
4. `Find_0_calibration_F1.sh`：只补充 calibration 的组件和 `F1_centered`。
5. `Find_0_validation_F1.sh` 与 `Find_0_train_F1.sh`：连续产生完整图概率、组件和 `F1_centered`，不计算 CLG。

以后需要 CLG 时，在相同正式产物根目录运行三个 `*_CLG.sh`。这些脚本使用原有 `*-f1-clg` 命令；已经完成的 probability、components 和 `F1_centered` 会按完成标记跳过，只增加 `CLG_centered`。重复运行同一分片不会重写已经完成的文件。

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

# 用两张可用 GPU 完成两个 calibration probability 分片；提交前设置当时可用的资源名。
gpu_resource="${GPU_RESOURCE:?请设置可用的 Slurm GPU 资源名，例如 a800、a100、h100 或 h200}"
bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
  --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_calibration_probability.sh \
  --resource "${gpu_resource}" --gpus 1 --cpus 8 \
  --array '0-1' \
  --after_hold \
  --job-name find0_cal_prob
```

`--after_hold` 表示每个数组元素结束后创建 `try_lock` 并保留资源；省略它时任务结束后自动释放。冻结阈值、calibration F1、validation F1 和 train F1 应分别提交；不要把后一个阶段设置为在前一个阶段尚未完成时启动。
