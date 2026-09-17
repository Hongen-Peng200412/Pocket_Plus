# Stage1 V3 推理提交入口

`stage1_v3.sh` 是五个阶段共用的官方入口：它激活 Pocket Plus 环境并调用 `python -m src.inference.cli`，不把完整推理固定成一个长命令。`unet_c1_sampling_comparison.sh` 只编排三种 `unet_c1` 采样模型已经冻结的 F1 blobs+basic 校准与 held-out 测试。`find1_real_receptor_evaluation.sh` 编排一套冻结 `Find_1` checkpoint 的基础打分与 Gaussian 打分；`find1_cryoatom2_receptor_evaluation.sh` 在相同阶段与 checkpoint 下改用 CryoAtom2 最终受体数据根；`find1_real_receptor_scored_centered.sh` 复用真实受体已冻结参数，为 Stage2 和 Stage3 生成 calibration/validation scored-centered 产物；`find1_real_receptor_train_shards.sh` 使用同一冻结身份为训练 PDB 的一个双卡四片批次生成 scored-centered 产物；`find1_real_receptor_train_shard_05.sh` 与 `find1_real_receptor_train_shard_06.sh` 在单卡、32 CPU allocation 中分别只生成用户第 5 片与第 6 片产物。所有编排脚本内部仍逐阶段调用 `stage1_v3.sh`。

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
- `--pdb-json`：顶层可以是字符串列表，也可以是通过 `pdb_ids` 字段保存字符串列表的对象；后者用于 held-out `test_0.json`。对象式清单的构造示例为 `{"pdb_ids":["1abc","2xyz"]}`。
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

- `--fit-semantic --data-root <Stage1数据根>`：读取当前完整 calibration 清单的 probability 与 `union_mask.npy`，拟合 PDB 等权 macro F-alpha 阈值，并把 `F{alpha}_semantic.json` 与 `F{alpha}_semantic_scan.npz` 写入 producer 的 `tuning/`。
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

正常 centered 需要 checkpoint、resolved config、模型代码来源和显式 `--forward-min-voxels`。该阈值按完整来源 blob 的体素数决定候选是否进入前向；`fits_centered_box=false` 的 blob 同样使用合法 80³ BOX 前向，并在产物中保留可容纳标志和完整来源体素数。选择参数 JSON 中的 `prefiltered_min_voxel` 与 `min_voxels` 只决定 `selected`，不会改变已经前向的候选集合。

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

所有 producer 都执行完整前向，并共同保存 `voxel_final`、辅助受体体素与三张 48³ 稠密数组。`Find_*` 另外保存 A/P 表。alpha 不改变字段集合。

若来源 `F{alpha}_blobs.npz:blob_index` 的长度严格大于 1000，当前 PDB 总是写
`status/F{alpha}_centered/_BLOB_EXCEED`。默认随后跳过 centered；显式增加
`--continue-on-blob-exceed` 时保留该提示标记并继续生成 centered。该开关不增加
覆盖、恢复、自动清理或独立状态管理机制。

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

`tune --score-mode basic` 直接读取 blobs，以来源平均概率为分数；候选事实仍包含 `fits_centered_box=false` 的 blob。命令显式给出的 `--prefiltered-min-voxel` 在任何参数尝试前固定，体素数低于该值的候选在全部组合中保持未入选。输出为 `tuning/F{alpha}_basic.json`。

`tune --score-mode gaussian` 对 Find centered 应用同一预过滤，再读取 A 原子表并依次执行 Gaussian 粗搜索、细搜索和最小体素数搜索。输出为 `tuning/F{alpha}_gaussian.json`。

两个模式都读取当前通过 `STAGE1_INFERENCE_CONFIG` 激活配置中的 `calibration.workers`；默认双卡配置为 56，A100 配置为 16，A800 配置为 24。该值同时用于候选/occurrence 文件读取、逐 PDB 事实构造和相互独立的参数目标计算；basic 的实际 float32 分数阈值扫描仍按降序串行累计。脚本把 OMP、MKL 与 OpenBLAS 内部线程固定为 1，由调参线程池提供外层并发。并发结果按 YAML 列表原顺序收集，目标并列时仍由原列表中的首项获胜。

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

`prefiltered_min_voxel` 与最终搜索出的 `min_voxels` 是两个独立门槛，不要求前者小于、等于或大于配置中的搜索值。最终 score-only 与 evaluate 同时应用两者。`objective_beta` 与 alpha 相互独立；未传 `--objective-beta` 时读取配置中的 2.0。调参目标是 semantic、coverage@0.3 和 one-to-one@0.3 三项 PDB 等权 macro F-beta 之和；三项 1:1:1 等权，任一 PDB 的局部分母为零时该项记为 0.0。

## 5. 独立评估

`evaluate` 必须显式提供 `--evaluation-name`，并从两种候选范围中选择一种：

- `--selection-parameters <json>`：按 JSON 中的 basic 或 Gaussian 参数重算 `score` 与 `selected`，只让 `selected=true` 的候选进入语义、真实侧覆盖、一对一匹配和 top-K 指标。有效组合是 blobs+basic、centered+basic 和 Find centered+Gaussian；Gaussian 需要 centered A 原子字段，不能用于 blobs。
- `--all-candidates`：不执行 basic 或 Gaussian 二次打分，把当前 blobs 或 centered 文件中的全部候选纳入指标；排序分数使用已有 `source_probability_mean`。centered 的“全部”只指已经进入 centered 文件的候选，不补回被 `forward_min_voxels` 或 `_BLOB_EXCEED` 排除的 blobs。

`evaluation-name` 直接成为 NPZ、JSONL 和 metrics JSON 的文件名主体。不同参数只要使用不同名称就能在同一输出目录并存，例如 `f2_centered_basic_recall` 与 `f2_centered_basic_precision`；代码不从参数内容生成摘要。

```bash
bash 训练与运行/sh/infer/stage1_v3.sh evaluate \
  --producer unet_c1 \
  --pdb-json /绝对路径/validation.json \
  --split validation \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2 \
  --alpha 2 \
  --artifact centered \
  --evaluation-name f2_centered_basic_selected \
  --selection-parameters /绝对路径/F2_basic.json \
  --data-root /storage/penghongen/AdaLigand/Ori_Data
```

不经过二次打分的 blobs 对照评估为：

```bash
bash 训练与运行/sh/infer/stage1_v3.sh evaluate \
  --producer unet_c1 \
  --pdb-json /绝对路径/validation.json \
  --split validation \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1_f2 \
  --alpha 2 \
  --artifact blobs \
  --evaluation-name f2_blobs_all \
  --all-candidates \
  --data-root /storage/penghongen/AdaLigand/Ori_Data
```

若 centered 因默认 `_BLOB_EXCEED` 行为而缺失，tune/evaluate 只在标准输出说明该原因并跳过该 PDB，不写另一套跳过状态。提示模式已经生成 centered 时，tune/evaluate 直接消费 centered，不读取 `_BLOB_EXCEED`。

`evaluate` 还会从每个实际完成候选评估的 PDB 完整图 `probability_map` 与 `union_mask` 发布 `semantic_micro_prauc` 和 `semantic_macro_prauc`。coverage 与 one-to-one 也在每个双向覆盖阈值下发布 micro/macro PRAUC：固定应用两个来源体素数门槛，忽略最终分数阈值，再按实际 float32 候选分数扫描。精确阈值、包含端点与阶梯积分公式见 `src/inference/README.md` 的评估字段说明。

## 三种 unet_c1 采样模型的固定入口

三模型比较固定使用 `alpha=1`、`objective_beta=1`、blobs+basic。occurrence-centric 复用既有 calibration 产物，只补做 held-out 测试；两个 pdb-centric 模型分别在 100 个 calibration PDB 上拟合语义阈值和 basic 参数，再评估同一 held-out `test_0.json`。正式命令只有：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" occurrence
```

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" pdb_centric_1
```

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" pdb_centric_2
```

occurrence 模式读取 `stage1_v3_a100_16cpu.yaml`，完整图 batch 为 12；两个 pdb-centric 模式读取 `stage1_v3_a800_24cpu.yaml`，完整图 batch 为 24。`stage1_v3.sh` 也接受环境变量 `STAGE1_INFERENCE_CONFIG` 选择其他同契约配置，未设置时仍使用 `stage1_v3.yaml`。

## Find_1 真实受体固定入口

`find1_real_receptor_evaluation.sh` 固定使用 Job `368455` 在 W&B 已完成步编号 `38622` 对应的 checkpoint。基础打分先在 100-PDB calibration 上拟合 F1 semantic，再以 `objective_beta=1` 调整 basic 参数；Gaussian 打分复用同一 probability，拟合 F2 semantic、执行 Find centered，再以 `objective_beta=1` 调整 Gaussian 参数。两种评分都完整评估 179-PDB `test_0`。149-PDB `test_1` 在该正式入口成功结束后，由执行记录中单独列出的任务临时命令从同一批逐 PDB 事实保序派生。

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/find1_real_receptor_evaluation.sh"
```

正式入口把 probability 与 centered 固定分成两个互斥 PDB 子序列，分别绑定 `CUDA_VISIBLE_DEVICES=0/1`。每个进程读取 `stage1_v3.yaml`，使用完整图 batch 18、centered batch 12 和每张 GPU 26 个请求物化线程；单进程 blobs 与 tune 使用 56 个外层线程。centered 显式传入 `--continue-on-blob-exceed`，因此候选数严格大于 1,000 时只保留提示标记，不排除该 PDB。

基础评估名为 `f1_blobs_basic_macro_selected`，Gaussian 评估名为 `f2_centered_gaussian_macro_selected`。长期正式入口不调用 `tmp/`。首个动态命令完成 `test_0` 并重新进入 `try_lock` 后，执行记录中的一次性派生命令才运行 `tmp/find1_real_receptor_evaluation_20260912/derive_test1.py`；`held_out_test_1/evaluation/` 只保存两套全局 JSON、逐 PDB JSONL 与 provenance，不重复保存 probability、blobs、centered 或逐 PDB evaluation NPZ。

### Find_1 CryoAtom2 受体入口

`find1_cryoatom2_receptor_evaluation.sh` 使用上述同一 `Find_1` checkpoint、calibration 清单、`test_0` 清单、F1 basic 与 F2 Gaussian 阶段顺序，但分别把 calibration 和 `test_0` 路由到已适配的 CryoAtom2 `Ori_Data`。它从 calibration 独立选择语义阈值和候选打分参数，不复用真实受体调参结果；F2 Gaussian 参数冻结后，使用 `centered --score-only` 非破坏性增补 calibration 的 `score/selected`。

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/find1_cryoatom2_receptor_evaluation.sh"
```

calibration 数据根为 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/Ori_Data`，`test_0` 数据根为 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data`。双 H100 使用同一有序 PDB 清单的两个互斥分片；完整图 batch 18、centered batch 12、每个 GPU 进程 26 个请求物化线程，单进程 tune 使用 56 个外层线程。入口保持 `forward_min_voxels=8` 和 `--continue-on-blob-exceed`，只完整评估 179-PDB `test_0`；149-PDB `test_1` 由执行记录中与正式入口分开的一次性命令保序派生。

### Stage2、Stage3 scored-centered 入口

`find1_real_receptor_scored_centered.sh` 只使用上述 calibration 冻结的 F2 语义概率阈值和 `objective_beta=1` Gaussian 选择参数。calibration 不重新前向：它通过 `centered --score-only` 向已有 `F2_centered.npz` 替换 `score` 和 `selected`，其他数组、候选顺序与 offsets 保持不变。validation 使用 `stage1_preparation_box_pool_3/split/pdb_split/validation.json`，依次生成 probability、冻结阈值 F2 blobs、centered，然后以同一组 Gaussian 参数写入 `score/selected`。该入口不执行 tune 或 evaluate，不访问 validation 标签。

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/find1_real_receptor_scored_centered.sh"
```

两个 GPU 分片继续使用完整图 batch 18、centered batch 12，每个 GPU 进程使用 26 个请求物化线程。`forward_min_voxels=8` 与前一阶段保持一致；`--continue-on-blob-exceed` 保证 blob 数严格大于 1,000 时仍保留该 PDB 并执行 centered 前向。产物继续位于 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/{calibration,validation}`。

### 训练 PDB 固定 50 片入口

三个训练分片入口均使用 `stage1_preparation_box_pool_3/split/pdb_split/train.json`。片身份与 `src.inference.cli` 相同：先以 `random.Random(3407)` 打乱完整 PDB 清单，再以 `[shard_index::50]` 取片；入口把用户可读片号传给 CLI 前严格减一。双卡入口的四个位置依次是 GPU 0 的两个片号和 GPU 1 的两个片号，下面第一条命令因此让 GPU 0 顺序处理第 1、2 片，让 GPU 1 顺序处理第 3、4 片。两个单卡入口分别只接受用户片号 5 和 6，对应 CLI `shard-index=4` 与 `shard-index=5`。

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/find1_real_receptor_train_shards.sh" 1 2 3 4
```

```bash
exec bash "/absolute/frozen/Pocket_Plus_release/训练与运行/sh/infer/find1_real_receptor_train_shard_05.sh" 5
```

```bash
exec bash "/absolute/frozen/Pocket_Plus_release/训练与运行/sh/infer/find1_real_receptor_train_shard_06.sh" 6
```

双卡入口必须恰好给出四个互不重复的片号；两个单卡入口分别只接受用户片号 5 或 6。所有入口对每片都依次执行 probability、冻结 F2 阈值 blobs、`forward_min_voxels=8` centered 和冻结 Gaussian score-only。两个单卡入口还会在任何产物续写前核验训练 PDB 清单、checkpoint、已解析训练配置 `config.yaml`、`F2_semantic.json` 和 `F2_gaussian.json` 的冻结 SHA-256；任一文件缺失或 SHA-256 与脚本中的固定摘要不一致时，以退出码 2 终止。双卡入口的两个 GPU 阶段进程各使用 26 个请求物化线程；同一对分片的 blobs 串行使用 56 个 worker，避免在 64 CPU allocation 内同时建立 112 个 blobs 线程。单卡入口固定使用 allocation 内的 CUDA device 0，读取 `stage1_v3_h100_32cpu.yaml`：完整图 batch 为 18，centered batch 为 12，两阶段各使用 26 个请求物化线程，blobs 使用 30 个 worker。跨项目接管时，allocation 注入的 `TASK_PROJECT_ROOT` 仍属于原项目，因此动态命令必须以冻结 Pocket Plus release 的绝对路径调用单卡入口，并在重试时复用同一 release。这些入口都不执行 tune、evaluate 或 overwrite。probability、F2 blobs 和不带 `--score-only` 的 centered 完整前向已有完成标记时直接跳过；score-only 仍幂等重算 `score/selected`，并通过同目录临时文件原子替换 `F2_centered.npz` 和对应完成标记。正式产物位于 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train/<pdb_id>`。

## 发布与复用

NPZ、JSON 和 JSONL 先写同目录临时文件，再以 `os.replace` 原子替换。角色 `_COMPLETE` 只表示当前文件发布完成，不保存 checkpoint、配置、代码摘要或哈希。同一 `output_root` 可以复用 probability 并逐次增加多个 alpha；如果同一 checkpoint 的 F3 最优与 F2 最优属于两个科学版本，由调用者使用两个易读目录区分，不由代码推断版本身份。

正式任务仍经 `训练与运行/submit_task.sh` 提交，以保存 release、launch、资源和 Slurm 日志。`stage1_v3.sh` 本身不申请资源，也不操作锁。

# Emap2lig 官方 Find held-out 入口

`emap2lig_official_find_li.sh` 依次执行 Emap2lig v0.3.4 官方 Find 和 Pocket Plus
标准 Stage1 评估。正式前向只覆盖 179-PDB `test_0`；149-PDB `test_1` 从同一批
逐 PDB 交集、匹配和语义 PRAUC 事实保序派生，不重复模型前向。

该入口固定使用官方 Li 阈值、官方少于 32 体素过滤和 detection batch size 16。
每个官方保留实例都进入指标，分数为实例在 Emap2lig 原生 ligand probability
中的平均概率。概率图以线性插值映射到 Pocket Plus V3 网格，实例以最近体素
映射并保留原官方编号；不同实例映射后可以重叠。

正式命令：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/emap2lig_official_find_li.sh"
```
