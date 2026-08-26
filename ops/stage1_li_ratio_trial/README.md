# Stage1 Li 阈值与候选比例实验

本目录在 `unet_c1` 的 100 个 calibration PDB 上拟合 Li basic_ratio 参数，再
冻结参数评估 200 个 validation PDB。实验复用已经发布的完整图
`probability_map.npz`，不恢复历史 component forest，不生成 centered，不加载
checkpoint，也不申请 GPU。

当前实验代码位于独立工作树和独立分支，全部改动保持 unstaged。实验结果决定
后续是否吸收实现；在用户明确允许前，不创建 Git 提交。

## 阅读顺序

1. [PLAN.md](PLAN.md)：说明已经冻结的科学规则、实施阶段与验收条件。
2. 本文件：说明入口、参数、输入和产物字段。
3. [EXECUTION.md](EXECUTION.md)：只记录实际发生的同步、提交、作业和验收事件。

## 科学规则

### Li 概率阈值

每个 PDB 独立计算一次 Li 最小交叉熵阈值：

1. 把完整概率图转为 `float64`，以全图均值初始化阈值。
2. 每轮把 `probability <= threshold` 归入背景，把
   `probability > threshold` 归入前景。
3. 使用两侧均值更新阈值；容差固定为 `1e-5`，最多更新 100 次。
4. 把原始阈值向上量化到分母为 32768 的概率网格。
5. 以 `probability >= li_threshold_applied` 提取全部 26 邻域连通区域。

实验不按体素数删除 Li 连通区域。`prefiltered_min_voxel=8` 只在后续候选比例
调参开始前生效，最终 `min_voxels` 仍按 `stage1_v3.yaml` 的原列表独立搜索。

### `score_ratio_threshold`

每个 PDB 先用 `prefiltered_min_voxel=8` 建立固定候选总体。候选按
`source_probability_mean` 降序排列；分数并列时保持 `Li_blobs.npz` 的候选顺序。
若固定总体含 `N` 个候选，比例为 `r`，则保留数量是：

```text
K = floor(N * r + 0.5)
```

因此，第 `k` 个候选在比例 `(k - 0.5) / N` 进入选择集合。调参不使用固定
比例网格，而是把 100 个 PDB 的全部候选数变化点组成一个精确有理数轴，并逐个
状态计算目标。目标并列时保留较小比例。冻结比例后，再把现有
`calibration.min_voxel_values` 依次作为最终后过滤条件搜索；该后过滤不改变
`N` 或 `K`。

F1 与 F2 分别采用 `objective_beta=1` 和 `objective_beta=2`。两者都最大化以下
PDB 等权 macro 三项目标之和：

1. semantic macro F-beta；
2. coverage@0.3 macro F-beta；
3. one-to-one@0.3 macro F-beta。

三项权重为 1:1:1。evaluate 仍同时发布 semantic、coverage 和 one-to-one 的
micro/macro F1、F2，以及完整图 semantic micro/macro PRAUC。

## 输入与服务器隔离

正式来源根目录是：

```text
/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950
```

实验唯一目标根目录是：

```text
/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li
```

calibration 和 validation 两项任务分别用 `rsync -a` 真实复制以下来源，不使用
软链接或硬链接：

- `inputs/` 中的 calibration 清单和现有推理配置；
- calibration 100 个和 validation 200 个 PDB 的完整 `probability/` 目录，包括
  概率 NPZ 与几何记录；
- 同一批 PDB 的完整 `status/probability/` 目录，包括完成标记与性能记录。

旧 blobs、centered、tuning 和 evaluation 不进入目标根目录。脚本不使用
`rsync --delete`，也不修改来源根目录。

## 目录结构

实验产物与后续正常 Stage1 推理保持同构：

```text
unet_c1-mainchain-ligand_PRAUC_0.602950--Li/
├── inputs/
├── artifacts/
│   └── unet_c1/
│       ├── tuning/
│       │   ├── Li_F1_basic_ratio.json
│       │   └── Li_F2_basic_ratio.json
│       ├── calibration/
│       │   ├── <pdb_id>/
│       │   │   ├── probability/probability_map.npz
│       │   │   ├── blobs/Li_blobs.npz
│       │   │   ├── status/probability/_COMPLETE
│       │   │   ├── status/Li_blobs/_COMPLETE
│       │   │   └── evaluation/
│       │   └── evaluation/
│       │       ├── li_f1_blobs_basic_ratio_selected.jsonl
│       │       ├── li_f1_blobs_basic_ratio_selected.metrics.json
│       │       ├── li_f2_blobs_basic_ratio_selected.jsonl
│       │       └── li_f2_blobs_basic_ratio_selected.metrics.json
│       └── validation/
│           ├── <pdb_id>/
│           │   ├── probability/probability_map.npz
│           │   ├── blobs/Li_blobs.npz
│           │   ├── status/probability/_COMPLETE
│           │   ├── status/Li_blobs/_COMPLETE
│           │   └── evaluation/
│           └── evaluation/
│               ├── li_f1_blobs_basic_ratio_selected.jsonl
│               ├── li_f1_blobs_basic_ratio_selected.metrics.json
│               ├── li_f2_blobs_basic_ratio_selected.jsonl
│               └── li_f2_blobs_basic_ratio_selected.metrics.json
├── feedback/
└── monitoring/
```

`tuning/` 只保存 producer 级选择参数；逐 PDB 产物与数据划分评估继续位于
`calibration/` 或 `validation/`。只有 calibration 搜索参数；validation 直接读取
并冻结复用两份 `Li_F*_basic_ratio.json`。本实验不会创建 centered 目录。

## `Li_blobs.npz` 新字段

`Li_blobs.npz` 保留 `extract_probability_blobs()` 的全部标准字段，并增加：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `li_threshold_raw` | `float32 (1,)` | Li 迭代得到的原始阈值 |
| `li_threshold_grid_index` | `int32 (1,)` | 向上量化到 1/32768 网格后的编号 |
| `li_threshold_applied` | `float32 (1,)` | 实际提取连通区域的包含端点阈值 |

标准 `source_threshold_value` 与 `li_threshold_applied` 表达同一实际阈值。

## 选择 JSON

`Li_F1_basic_ratio.json` 与 `Li_F2_basic_ratio.json` 的共同核心字段是：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `candidate_role` | 字符串 | 固定为 `Li_blobs` |
| `threshold_method` | 字符串 | 固定为 `li` |
| `li_denominator` | 整数 | 固定为 32768 |
| `objective` | 浮点数 | 最终 macro 三项目标之和 |
| `objective_beta` | 浮点数 | F1 为 1，F2 为 2 |
| `score_mode` | 字符串 | 固定为 `basic_ratio` |
| `score_parameters` | JSON 对象 | 空对象；候选分数就是来源区域平均概率 |
| `score_ratio_threshold` | 浮点数 | 每个 PDB 固定预过滤总体的保留比例 |
| `prefiltered_min_voxel` | 整数 | 固定为 8 |
| `min_voxels` | 整数 | 冻结比例后搜索得到的最终来源体素数下限 |
| `stages` | JSON 对象 | 保存精确比例扫描和最终体素数搜索结果 |

实验 JSON 不保存 `score_threshold`，不包含 SHA、checkpoint 身份或额外状态字段。
不同科学配置继续由清晰的外层结果目录区分。

## 本地验证

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest -q \
  tests/inference/test_stage1_li_ratio_trial.py \
  tests/inference/test_stage1_v3.py \
  tests/inference/test_calibration_parallel.py
```

专项测试覆盖历史 Li 数值规则、向上量化、half-up 数量、固定预过滤总体、精确
比例状态、比例并列规则、串并行一致性，以及同时含 micro/macro/PRAUC 的小型
端到端发布。

## 服务器入口

正式任务脚本是 `run_split.sh`。提交时必须显式传入数据划分、来源根、实验目标根
和 AdaLigand 数据根；脚本在 allocation 内完成真实复制和全部 CPU 计算。
calibration 命令是：

```bash
bash 训练与运行/submit_task.sh \
  --task-root /home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial \
  --sh ops/stage1_li_ratio_trial/run_split.sh \
  --resource cpu \
  --cpus 16 \
  --mem 64G \
  --job-name unet_c1_li_ratio_calibration \
  -- \
  calibration \
  /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950 \
  /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li \
  /storage/penghongen/AdaLigand/Ori_Data
```

calibration 验收通过后，validation 命令只把作业名改为
`unet_c1_li_ratio_validation`，并把任务参数中的 `calibration` 改为
`validation`。validation 不进入 `tune_centered_selection()`。

任务不使用 `pre_hold`、`after_hold`、`try_lock` 或 `kill_lock`。监视采用自然 shell
睡眠，每次间隔 10 至 30 分钟；不建立 heartbeat。
