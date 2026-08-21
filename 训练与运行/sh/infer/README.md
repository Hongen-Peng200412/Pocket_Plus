# Stage1 V3 推理提交入口

本目录只保留一个正式脚本 `stage1_v3.sh`。脚本激活 Pocket Plus 环境并调用 `python -m src.inference.cli`，不把完整推理固定成一个长命令。使用者可以分别提交完整图概率、连通区域、centered 特征、参数调优和评估五个阶段，也可以按相同顺序连续提交。

旧 `calibrate`、`run`、固定 `F1_basic`、固定 `F3_centered`、Selector、CLG、组件森林和 Li 入口已经退出活动代码树。需要考察旧行为时使用 Git 历史。

## 五个命令的关系

```text
probability
    ↓
blobs ───────────────→ tune --score-mode basic
    ↓
centered ────────────→ tune --score-mode gaussian
    ↓                         ↓
centered --score-only ← F{alpha}_basic.json 或 F{alpha}_gaussian.json
    ↓
evaluate --artifact blobs|centered
```

`alpha` 控制完整图语义 F-alpha 阈值和文件标签。`alpha=2.0` 对应 `F2_blobs.npz` 与 `F2_centered.npz`；`alpha=0.5` 对应 `F0p5_blobs.npz` 与 `F0p5_centered.npz`。配置文件中的推荐值是 2.0，命令可以用 `--alpha` 覆盖。

## 共同参数

五个命令都显式接收：

- `--producer`：产物目录名，例如 `unet_c1`、`unet_base`、`Find_0` 或未来新增的同类名称。CLI 不维护名称白名单。
- `--pdb-json`：顶层为字符串列表的 JSON 文件，例如 Stage1 V3 的 `validation.json`。
- `--split`：写入产物目录的数据划分名，例如 `calibration`、`validation` 或 `train`。
- `--output-root`：当前推理结果根目录。同一目录可以先保存 probability，再逐次增加多个 alpha 的 blobs 和 centered。

`probability`、使用显式阈值的 `blobs` 和 `centered` 可以增加 `--shard-count N --shard-index I`。CLI 先用固定 seed 3407 打乱完整 JSON 清单，再取 `[I::N]`；`I` 从 0 开始。相同 JSON、N 和 I 始终得到相同 PDB 子序列。语义拟合、tune 和 evaluate 必须读取完整清单，不分片。

## 1. 生成完整图概率

`probability` 是唯一需要 checkpoint 的完整图阶段。`--model-code-source` 必须显式选择 `current_workspace` 或 `training_snapshot`。

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/infer/stage1_v3.sh \
  --resource h100 --gpus 1 --cpus 16 \
  --job-name unet_c1_probability \
  -- probability \
  --producer unet_c1 \
  --checkpoint /绝对路径/checkpoints/TOP_epoch_03_score_0.4123.ckpt \
  --resolved-config /绝对路径/config.yaml \
  --model-code-source current_workspace \
  --pdb-json /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json \
  --split calibration \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2
```

默认遇到 `status/probability/_COMPLETE` 就跳过该 PDB。`--overwrite` 只撤销并重算 probability，不删除同一 PDB 已有的其他 alpha 产物。

## 2. 生成 F-alpha blobs

`blobs` 必须从以下三种阈值来源中选择一种：

- `--fit-semantic --data-root <Stage1数据根>`：读取当前完整 calibration 清单的 probability 与 `union_mask.npy`，拟合 micro F-alpha 阈值，并写 `F{alpha}_semantic.json` 与 `F{alpha}_semantic_scan.npz`。
- `--semantic-threshold <数值>`：直接使用显式概率阈值。
- `--semantic-parameters <JSON>`：读取另一条命令已经写出的 `threshold_value`。

```bash
bash 训练与运行/sh/infer/stage1_v3.sh blobs \
  --producer unet_c1 \
  --pdb-json /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json \
  --split calibration \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2 \
  --alpha 2 \
  --fit-semantic \
  --data-root /storage/penghongen/AdaLigand/Ori_Data
```

连通区域使用 26 邻域，保存阈值下的全部 blob，不在该阶段应用最小体素数。默认遇到对应 `status/F{alpha}_blobs/_COMPLETE` 就跳过；`--overwrite` 只重算该 alpha 的 blobs。

## 3. 生成 F-alpha centered

正常 centered 需要 checkpoint、resolved config、模型代码来源和显式 `--forward-min-voxels`。该阈值只决定哪些 `fits_centered_box=True` 的来源 blob 进入完整模型前向；选择参数 JSON 中的 `prefiltered_min_voxel` 与 `min_voxels` 只决定 `selected`，不会改变已经前向的候选集合。

```bash
bash 训练与运行/sh/infer/stage1_v3.sh centered \
  --producer unet_c1 \
  --checkpoint /绝对路径/checkpoints/TOP_epoch_03_score_0.4123.ckpt \
  --resolved-config /绝对路径/config.yaml \
  --model-code-source current_workspace \
  --pdb-json /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json \
  --split calibration \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2 \
  --alpha 2 \
  --forward-min-voxels 8
```

所有 producer 都执行完整前向并保存 `voxel_final`。`unet_*` 只保存 centered 共同字段与 `voxel_final`；`Find_*` 另外保存辅助受体体素、A/P 表和三张 48³ 稠密数组。alpha 不改变字段集合。

若来源 `F{alpha}_blobs.npz:blob_index` 的长度严格大于 1000，当前 PDB 立即写 `status/F{alpha}_centered/_BLOB_EXCEED` 并跳过 centered。该标记不拥有覆盖、恢复或自动清理机制。

传入 `--selection-parameters` 时，首次 centered 发布就增加 `score` 与 `selected`。已存在 centered 时也可以只在 CPU 上更新这两个字段：

```bash
bash 训练与运行/sh/infer/stage1_v3.sh centered \
  --producer unet_c1 \
  --pdb-json /绝对路径/validation.json \
  --split validation \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2 \
  --alpha 2 \
  --selection-parameters /绝对路径/F2_basic.json \
  --score-only
```

`--score-only` 不需要 checkpoint，不改变候选轴、offsets、几何、概率或特征。`--overwrite` 与 `--score-only` 互斥；前者完整重跑当前 centered 阶段。

## 4. 调整选择参数

`tune --score-mode basic` 直接读取 blobs，以来源平均概率为分数；候选事实仍包含 `fits_centered_box=false` 的 blob。命令显式给出的 `--prefiltered-min-voxel` 在任何参数尝试前固定，体素数低于该值的候选在全部组合中保持未入选。输出为 `calibration/F{alpha}_basic.json`。

`tune --score-mode gaussian` 对 Find centered 应用同一预过滤，再读取 A 原子表并依次执行 Gaussian 粗搜索、细搜索和最小体素数搜索。输出为 `calibration/F{alpha}_gaussian.json`。

```bash
bash 训练与运行/sh/infer/stage1_v3.sh tune \
  --producer Find_0 \
  --pdb-json /绝对路径/calibration.json \
  --split calibration \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/find0_f2 \
  --alpha 2 \
  --objective-beta 2 \
  --score-mode gaussian \
  --prefiltered-min-voxel 8 \
  --data-root /storage/penghongen/AdaLigand/Ori_Data
```

`prefiltered_min_voxel` 与最终搜索出的 `min_voxels` 是两个独立门槛，不要求前者小于、等于或大于配置中的搜索值。最终 score-only 与 evaluate 同时应用两者。`objective_beta` 与 alpha 相互独立；未传 `--objective-beta` 时读取配置中的 2.0。调参目标仍是 semantic、coverage@0.3 和 one-to-one@0.3 三项 micro F-beta 之和。

## 5. 独立评估

`evaluate` 显式选择 `--artifact blobs` 或 `--artifact centered`，并读取选择参数 JSON。有效组合是 blobs+basic、centered+basic 和 Find centered+Gaussian；Gaussian 需要 centered A 原子字段，不能用于 blobs。输出名同时编码 alpha、候选产物和打分模式，例如 `F2_blobs_basic` 或 `F2_centered_gaussian`。

```bash
bash 训练与运行/sh/infer/stage1_v3.sh evaluate \
  --producer unet_c1 \
  --pdb-json /绝对路径/validation.json \
  --split validation \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2 \
  --alpha 2 \
  --artifact centered \
  --selection-parameters /绝对路径/F2_basic.json \
  --data-root /storage/penghongen/AdaLigand/Ori_Data
```

若 centered 因 `_BLOB_EXCEED` 缺失，tune/evaluate 只在标准输出说明该原因并跳过该 PDB，不写另一套跳过状态。

## 发布与复用

NPZ、JSON 和 JSONL 先写同目录临时文件，再以 `os.replace` 原子替换。角色 `_COMPLETE` 只表示当前文件发布完成，不保存 checkpoint、配置、代码摘要或哈希。同一 `output_root` 可以复用 probability 并逐次增加多个 alpha；如果同一 checkpoint 的 F3 最优与 F2 最优属于两个科学版本，由调用者使用两个易读目录区分，不由代码推断版本身份。

正式任务仍经 `训练与运行/submit_task.sh` 提交，以保存 release、launch、资源和 Slurm 日志。`stage1_v3.sh` 本身不申请资源，也不操作锁。
