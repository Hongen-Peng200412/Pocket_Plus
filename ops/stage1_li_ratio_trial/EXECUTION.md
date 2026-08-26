# Stage1 Li 阈值与候选比例实验执行记录

本文件只记录已经发生的关键事件。科学规则和验收条件见 `PLAN.md`，入口和字段
说明见 `README.md`。

## 产物来源

正式 probability 来源根：

```text
/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950
```

服务器只读核对结果：来源根含 calibration 100 个
`probability/probability_map.npz` 和 100 个 `status/probability/_COMPLETE`，占用约
14 GiB。`inputs/calibration.json` 是 100 个小写 PDB 标识的 JSON 列表。

实验目标根：

```text
/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li
```

开始实现前，该目录存在且为空。

## 代码隔离

- 本地工作树：`C:\Users\15919\Desktop\Pocket_Plus_stage1_li_ratio_trial`
- 本地分支：`codex/stage1-li-ratio-calibration-trial`
- 基点：任务开始时的 `Learn/CUMULATIVE`
- 远端任务根：`/home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial`

本轮适用用户明确批准的实验例外：所有改动保持 unstaged，不执行 `git add`、
`git commit`、学习历史整理或 `Learn/CUMULATIVE` 移动。

## 已发生事件

### 2026-08-26：前置核对与本地实现

1. 核对主工作树只有既有 `训练与运行/sh/unet_c1.sh` 修改；实验工作树从
   `Learn/CUMULATIVE` 单独建立，没有触碰该文件。
2. 只读核对来源 probability 数量、输入清单和空目标目录。
3. 完成 Li 阈值、basic_ratio 精确比例搜索、显式评估角色和隔离实验入口。
4. 修改前基线命令：

   ```powershell
   D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest \
     tests/inference/test_stage1_v3.py -q
   ```

   结果：`32 passed in 4.83s`。

5. 第一轮实现后的定向命令：

   ```powershell
   D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest \
     tests/inference/test_stage1_li_ratio_trial.py \
     tests/inference/test_stage1_v3.py \
     tests/inference/test_calibration_parallel.py -q
   ```

   结果：`42 passed in 3.19s`。随后增加比例目标严格并列时保留较小比例的专项
   案例，并完成两遍逐函数自查。

6. 服务器提交前的完整 CPU 回归命令：

   ```powershell
   D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest \
     tests/inference/test_stage1_li_ratio_trial.py \
     tests/inference/test_stage1_v3.py \
     tests/inference/test_calibration_parallel.py \
     tests/datasets/test_stage1_dataset.py -q
   ```

   结果：`59 passed in 4.32s`。`run_calibration.sh` 同时通过 MSYS2
   `bash -n`；服务器对 rsync include 规则执行 dry-run，确认只会真实复制每个
   PDB 的完整 `probability/` 与 `status/probability/` 目录，不会复制旧 blobs、
   centered、tuning 或 evaluation。

### 2026-08-26：隔离同步与正式提交

1. 使用 MSYS2 `rsync -av` 把本地独立工作树非删除式同步到：

   ```text
   /home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial
   ```

   同步排除 `.git/`、`__pycache__/`、`.pytest_cache/` 和 `*.pyc`，没有使用
   `--delete`。Windows 工作树根的 `.git` 是普通指针文件，首次同步未被
   `.git/` 目录规则排除；发现后先解析并确认远端绝对任务根，再只删除该指针
   文件。远端任务根随后确认不含 Git 元数据，项目代码和实验目标数据均未删除。

2. 远端使用 `Pocket_Plus_centos7_cu121_allgpu` 环境执行 compileall、
   `bash -n` 和与本地相同的 59 项 CPU 回归，结果为
   `59 passed in 30.07s`。

3. 正式提交命令：

   ```bash
   bash /home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial/训练与运行/submit_task.sh \
     --task-root /home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial \
     --sh ops/stage1_li_ratio_trial/run_calibration.sh \
     --resource cpu \
     --cpus 16 \
     --mem 64G \
     --job-name unet_c1_li_ratio_calibration \
     --feedback-root /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li/feedback/task_runtime \
     -- \
     /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950 \
     /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li \
     /storage/penghongen/AdaLigand/Ori_Data
   ```

   Slurm 返回 Job `356954`。资源为 16 CPU、64 GiB、0 GPU，没有 pre-hold 或
   after-hold。首次检查时作业在 `cnode01` 运行，已经建立 release 与 launch，
   并开始把 calibration probability 真实复制到 Li 目标根。

### 2026-08-26：Li calibration 完成与验收

Job `356954` 最终为 `COMPLETED/0:0`，墙钟时间 10 分 21 秒，Slurm 记录的
MaxRSS 为 9,019,688 KiB。关键阶段结果是：

- 100 个 probability 与 100 个 probability `_COMPLETE` 真实复制完成；
- 100 个 PDB 全部发布 `Li_blobs/_COMPLETE`，合计 5,575 个 Li 连通区域；
- F1/F2 各有 100 个逐 PDB evaluation NPZ 和 100 行 JSONL；
- producer 树中 centered 目录数为 0；
- 两份 tuning JSON 都含 `score_ratio_threshold`，都不含 `score_threshold`。

F1 basic_ratio 的冻结结果：

| 字段 | 数值 |
| --- | ---: |
| `score_ratio_threshold` | 0.33912248628884833 |
| `min_voxels` | 37 |
| macro 三项目标 | 1.031946657074557 |
| semantic macro F1 | 0.3354504384663223 |
| coverage@0.3 macro F1 | 0.34870913195337944 |
| one-to-one@0.3 macro F1 | 0.3477870866548555 |

F2 basic_ratio 的冻结结果：

| 字段 | 数值 |
| --- | ---: |
| `score_ratio_threshold` | 0.7596153846153847 |
| `min_voxels` | 37 |
| macro 三项目标 | 1.1852500915140487 |
| semantic macro F2 | 0.42231569398586266 |
| coverage@0.3 macro F2 | 0.382056586264067 |
| one-to-one@0.3 macro F2 | 0.38087781126411857 |

两项 calibration 评估读取同一完整概率图，因此
`semantic_micro_prauc=0.5433163278270201`，
`semantic_macro_prauc=0.40964173927021513`。完整 micro/macro 候选指标保存在
Li 目标根的两份 `.metrics.json` 中。

### 2026-08-26：增加 Li validation 阶段

用户在 calibration 完成后把实验范围扩展到 200 个 validation PDB。实现调整为
统一 `run_trial.py` 与 `run_split.sh`：validation 为每个 PDB 计算自己的 Li
阈值，但只读取并冻结复用上述两份 calibration basic_ratio JSON，不执行参数
搜索。小型端到端测试同时覆盖 calibration 和 validation，并证明 validation
执行前后 tuning JSON 文本不变；完整 CPU 回归结果为 `59 passed in 3.80s`，
新 `run_split.sh` 通过 `bash -n`。

隔离工作树再次非删除式同步到同一远端任务根，并只删除该隔离任务根中已经被
统一入口替代的 `run_calibration.py` 和 `run_calibration.sh`。远端新入口随后通过
59 项回归，结果为 `59 passed in 7.35s`。validation 正式提交命令是：

```bash
bash /home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial/训练与运行/submit_task.sh \
  --task-root /home/penghongen/My_Project/Pocket_Plus_stage1_li_ratio_trial \
  --sh ops/stage1_li_ratio_trial/run_split.sh \
  --resource cpu \
  --cpus 16 \
  --mem 64G \
  --job-name unet_c1_li_ratio_validation \
  --feedback-root /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li/feedback/task_runtime \
  -- \
  validation \
  /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950 \
  /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li \
  /storage/penghongen/AdaLigand/Ori_Data
```

Slurm 返回 Job `356955`。首次检查时作业在 `cnode01` 运行，并开始真实复制 200
个 validation probability。

### 2026-08-26：Li validation 完成与验收

Job `356955` 最终为 `COMPLETED/0:0`，墙钟时间 27 分 59 秒，Slurm 记录的
MaxRSS 为 18,752,316 KiB。作业先真实复制 probability，再逐 PDB 生成 Li
blobs，最后冻结使用 calibration 选出的两套 basic_ratio 参数。数量核对结果为：

- 200 个 PDB 目录、200 个 `probability_map.npz` 和 200 个 probability
  `_COMPLETE`；
- 200 个 `Li_blobs.npz` 和 200 个 `Li_blobs/_COMPLETE`，合计 14,134 个 Li
  连通区域；
- F1/F2 各有 200 个逐 PDB evaluation NPZ 和 200 行 JSONL；
- validation 没有生成新的 tuning 文件，仍读取 calibration 的
  `Li_F1_basic_ratio.json` 与 `Li_F2_basic_ratio.json`。

F1 冻结参数为 `score_ratio_threshold=0.33912248628884833`、
`min_voxels=37`。validation 主要指标为：

| 指标 | 数值 |
| --- | ---: |
| semantic macro F1 | 0.3732883090779028 |
| coverage@0.3 macro F1 | 0.39895597349767414 |
| one-to-one@0.3 macro F1 | 0.39445430333578346 |
| semantic micro F1 | 0.43939077585554037 |

F2 冻结参数为 `score_ratio_threshold=0.7596153846153847`、
`min_voxels=37`。validation 主要指标为：

| 指标 | 数值 |
| --- | ---: |
| semantic macro F2 | 0.4588127774351583 |
| coverage@0.3 macro F2 | 0.4247253149365731 |
| one-to-one@0.3 macro F2 | 0.41949993028291144 |
| semantic micro F2 | 0.5716113648753791 |

两项 validation 评估读取同一完整概率图，因此
`semantic_micro_prauc=0.5080070028769534`，
`semantic_macro_prauc=0.4572529596856525`。完整 micro/macro 指标保存在目标根
`artifacts/unet_c1/validation/evaluation/` 下的两份 `.metrics.json` 中。

### 2026-08-26：联合分析与 handoff

`ANALYSIS.md` 已把 Li-F1、Li-F2 与经典 F1、经典 F2、经典 F2-beta1 在两个
数据划分上的 macro 三项目标、候选规模、top-3 与 PRAUC 对齐比较。Li-F1 与
Li-F2 的对应 macro 目标均低于经典方案；经典 F2-beta1 在 validation 上与经典
F1 的 F1 三项目标近乎相同，并取得最高 F2 三项目标。

隔离实验 handoff 写入
`CLAUDE/memory/handoffs/2026-08-26-stage1-li-threshold-ratio-trial.md`。所有 Li
代码、测试、文档和 handoff 继续只留在当前隔离工作树，保持 unstaged，没有
新增 commit。
