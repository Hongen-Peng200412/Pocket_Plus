# F2 语义 blobs 的 beta=1 basic 增量实验执行记录

本文件实现 `PLAN.md`，只记录已经发生的关键事件。当前实验只覆盖
`unet_c1-mainchain-ligand_PRAUC_0.602950` 的 F2 blobs，不改变 Stage1 正式科学
契约。

## 产物来源

正式结果根：

```text
/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950
```

2026-08-26 服务器只读核对结果：

- calibration 有 100 个 PDB 目录和 100 个 `F2_blobs.npz`；
- validation 有 200 个 PDB 目录和 200 个 `F2_blobs.npz`；
- `tuning/F2_basic.json` 是 `alpha=2.0`、`objective_beta=2.0`、
  `prefiltered_min_voxel=8`，绝对分数阈值为 0.3140856623649597；
- 两个数据划分已经有 `f1_blobs_basic_macro_selected` 与
  `f2_blobs_basic_macro_selected` 评估，本实验使用新名称并存。

## 代码边界

主工作树的实现端点 `a6208c7` 与 `Learn/CUMULATIVE` 端点 `21c61cf` 文件树
完全相同。开始本增量实验时，主工作树另有既存的
`训练与运行/sh/unet_c1.sh` 修改；本目录不读取、不修改也不暂存该文件。

用户明确要求本实验日志与 handoff 留在主工作树，并保持 unstaged。未经后续
明确允许，本轮不执行 `git add`、`git commit`、分支移动或学习历史整理。

## 2026-08-26：实现与本地验证

新增 `run_trial.sh`，只编排现有 `stage1_v3.sh tune/evaluate`。调参阶段在 Slurm
临时目录映射正式 calibration 目录，使现有 CLI 的默认
`F2_basic.json` 不接触正式同名文件；新选择结果随后在正式 tuning 目录内原子
发布为 `F2_basic_beta1.json`。脚本再使用该文件依次评估 calibration 和
validation。

两遍自查分别核对了脚本的职责、直线调用关系、分支嵌套和注释，以及每个路径
变量、alpha/beta、固定预过滤门槛、最终搜索门槛与输入清单的科学语义。脚本
没有新增 Python 函数或修改正式推理接口。

本地语法检查命令：

```powershell
C:\msys64\usr\bin\bash.exe -n ops/stage1_f2_beta1_trial/run_trial.sh
```

结果：通过。

正式推理 CPU 回归命令：

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest \
  tests/inference/test_stage1_v3.py \
  tests/inference/test_calibration_parallel.py \
  tests/datasets/test_stage1_dataset.py -q
```

结果：`54 passed in 3.75s`。`git diff --check` 通过，暂存区为空。

## 2026-08-26：隔离同步与正式提交

本地主工作树使用 MSYS2 `rsync -a` 非删除式同步到新的独立远端任务根：

```text
/home/penghongen/My_Project/Pocket_Plus_stage1_f2_beta1_trial
```

同步排除 `.git`、缓存目录和 `*.pyc`，结果为 677 个普通文件、约 103 MB，
删除文件数为 0。远端任务根不含 Git 元数据；脚本通过 `bash -n`，同一组正式
推理 CPU 回归结果为 `54 passed in 7.59s`。

正式提交命令：

```bash
bash /home/penghongen/My_Project/Pocket_Plus_stage1_f2_beta1_trial/训练与运行/submit_task.sh \
  --task-root /home/penghongen/My_Project/Pocket_Plus_stage1_f2_beta1_trial \
  --sh ops/stage1_f2_beta1_trial/run_trial.sh \
  --resource cpu \
  --cpus 16 \
  --mem 64G \
  --job-name unet_c1_f2_blobs_basic_beta1 \
  --feedback-root /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950/feedback/task_runtime \
  -- \
  /storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950 \
  /storage/penghongen/AdaLigand/Ori_Data
```

Slurm 返回 Job `356956`。资源为 16 CPU、64 GiB、0 GPU，没有 pre-hold 或
after-hold。原始阶段输出写入正式实验根的 `feedback/F2_basic_beta1.log`。

## 2026-08-26：正式运行完成与验收

Job `356956` 最终为 `COMPLETED/0:0`，墙钟时间 9 分 47 秒，Slurm 记录的
MaxRSS 为 1,408,252 KiB。calibration 调参选出：

| 字段 | 数值 |
| --- | ---: |
| `alpha` | 2.0 |
| `objective_beta` | 1.0 |
| `score_threshold` | 0.5382110476493835 |
| `prefiltered_min_voxel` | 8 |
| `min_voxels` | 24 |
| macro 三项目标 | 1.3114210328347076 |

新文件已经发布为 `artifacts/unet_c1/tuning/F2_basic_beta1.json`。原
`F2_basic.json` 仍为 `objective_beta=2.0`、`score_threshold=0.3140856623649597`
与 `min_voxels=21`，没有被本实验改写。

calibration 验收结果为 100 个逐 PDB NPZ 和 100 行 JSONL：

| 指标 | 数值 |
| --- | ---: |
| semantic macro F1 | 0.3975462836898636 |
| coverage@0.3 macro F1 | 0.4593467427702887 |
| one-to-one@0.3 macro F1 | 0.4545280063745551 |
| semantic micro F1 | 0.5556680840117771 |
| semantic macro PRAUC | 0.40964173927021513 |
| semantic micro PRAUC | 0.5433163278270201 |

validation 验收结果为 200 个逐 PDB NPZ 和 200 行 JSONL：

| 指标 | 数值 |
| --- | ---: |
| semantic macro F1 | 0.45529405612884943 |
| coverage@0.3 macro F1 | 0.5077642403872431 |
| one-to-one@0.3 macro F1 | 0.5011118112863627 |
| semantic micro F1 | 0.5189836737680014 |
| semantic macro PRAUC | 0.4572529596856525 |
| semantic micro PRAUC | 0.5080070028769534 |

两份 `.metrics.json` 还完整保存 macro/micro F2、0.5/0.6 覆盖阈值和 top-K
指标。联合对照分析写在隔离 Li 工作树的
`ops/stage1_li_ratio_trial/ANALYSIS.md`，避免把 Li 执行日志复制到主工作树。

## 收口

通用任务入口已为 Job `356956` 保存 release、launch、实际 `run_cmd.sh` 以及
Slurm 的 `out/err`；本执行记录另行保存了用户可直接复用的完整提交命令、数据
来源和结果。主工作树 handoff 写入
`CLAUDE/memory/handoffs/2026-08-26-stage1-f2-blobs-beta1-trial.md`。

本实验新增文件、执行记录、项目状态与 handoff 均保持 unstaged，没有执行
commit。开始任务前已有的 `训练与运行/sh/unet_c1.sh` 修改仍由用户保有。
