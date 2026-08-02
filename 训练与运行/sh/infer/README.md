# Find_0 正式推理入口

本目录保存 Find_0 的正式推理脚本。脚本不依赖隔离 smoke 的代码或产物；它们从训练运行目录读取已经冻结的 checkpoint 与配置，并把全部正式产物写入：

`/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/`

## 固定模型

- checkpoint：`/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-Find_0-CPC1/Find_0-CPC1____job321743_Find_0_CPC1_lr5e5_p2_val30_chunk2x_2gpu_m8_w1/checkpoints/TOP_epoch_00_score_0.2843.ckpt`
- checkpoint SHA-256：`87ec6080809a14e2558e10c0740c16371b363fed0f672081b389f46c64928b44`
- 训练配置：同一运行目录下的 `config.yaml`
- 训练配置 SHA-256：`3eaae2769bdc7df4f0ddfcd40ac5fd34d4d71a9f5119a1f300dd24ea11b82fcf`
- 选择依据：该 checkpoint 对应的 `val_score/global/voxel_ligand_PRAUC` 为 `0.6754766702651978`，是现存 Find_0 checkpoint 中的最高值。

checkpoint 的 `src_snapshot/src` 提供模型与 wrapper 定义；当前 release 提供推理编排、Dataset 和产物读写代码。隔离 smoke 已用真实 checkpoint、真实配置和 A–G 数据验证这套组合。当前推理代码只做了三个兼容性补齐：模型加载前不提前导入 Dataset；组件生成前先激活 checkpoint 代码快照；无监督推理批次为旧 Find 伪原子注入补充不参与得分的布尔型全零 `atom_label`。

## 正式执行顺序

1. `Find_0_prepare_inputs.sh`：从已经过滤的 Stage1 集合文件提取、排序并冻结唯一 PDB 清单，同时核对源文件、checkpoint 和配置的 SHA-256。
2. `Find_0_calibration_probability_A800.sh`：按分片产生 calibration 完整图概率。
3. `Find_0_freeze_thresholds.sh`：在完整 calibration 集合上冻结阈值，并写出语义与实例评估结果；`min_voxels=15`、`max_voxels=2046`、阈值网格分母为 `32768`。
4. `Find_0_calibration_F1_A800.sh`：只补充 calibration 的组件和 `F1_centered`。
5. `Find_0_validation_F1_A800.sh` 与 `Find_0_train_F1_A800.sh`：连续产生完整图概率、组件和 `F1_centered`，不计算 CLG。

以后需要 CLG 时，在相同正式产物根目录运行三个 `*_CLG_A800.sh`。这些脚本使用原有 `*-f1-clg` 命令；已经完成的 probability、components 和 `F1_centered` 会按完成标记跳过，只增加 `CLG_centered`。重复运行同一分片不会重写已经完成的文件。

## 分片与资源

每个 A800 脚本顶部都明确写有 `global_shard_count=16`，并从 `SLURM_ARRAY_TASK_ID` 读取当前 `shard_index`。例如，申请八张 A800、但只运行前八个全局分片时，提交数组可以写成 `0-7`；稍后再提交 `8-15` 即可补齐其余分片。不同显卡脚本可以共同写入同一正式目录，但分片编号不得重叠。

隔离 A800 smoke 使用完整图批量 8，并完成 calibration、validation 和 train 的完整图推理。centered 批量 12 虽然成功，但最高观察到约 81.2/81.9 GiB，因此正式 A800 脚本保守使用 centered 批量 8。当前没有把未经 smoke 的 A100 参数写成正式入口。

正式任务由 `训练与运行/submit_task.sh` 提交。该入口接受本目录脚本的绝对路径或项目内相对路径，并在 allocation 中建立 release、launch 记录和四锁控制。提交命令必须在人工核对脚本与当前可用显卡后单独执行；本目录脚本本身不会申请资源。

以下命令只是待人工确认的提交模板，不会由任何推理脚本自动调用：

```bash
# 先冻结三份唯一 PDB 清单和模型来源记录。
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/infer/Find_0_prepare_inputs.sh \
  --resource cpu --cpus 4 \
  --job-name find0_infer_inputs

# 示例：用八张 A800 并发完成 16 个 calibration probability 分片；数组会分两批调度。
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/infer/Find_0_calibration_probability_A800.sh \
  --resource a800 --gpus 1 --cpus 8 \
  --array '0-15%8' \
  --job-name find0_cal_prob
```

冻结阈值、calibration F1、validation F1 和 train F1 应分别提交；不要把后一个阶段设置为在前一个阶段尚未完成时启动。正式执行前还要根据当时的空闲显卡决定数组并发上限。
