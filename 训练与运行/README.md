# 训练与运行

本目录提供一套通用的 Slurm 任务提交方式，以及三份可以直接阅读和修改的
AdaLigand Stage1 正式训练脚本。

日常使用只需要理解以下三层：

```text
submit_task.sh
→ sbatch/task.sbatch
→ sh/<具体训练>.sh
```

- `submit_task.sh`：选择任务、GPU 类型、GPU 数、CPU 数和可选锁。
- `sbatch/task.sbatch`：唯一真正交给 `sbatch` 的通用资源包装脚本。
- `sh/Find_0.sh`、`sh/Find_1.sh`、`sh/unet_c1.sh`：直接决定具体训练
  使用的数据、环境、experiment、Hydra 覆盖和正式产物位置。

release、launch 与四锁循环的实现位于
`与服务器交互/other/training_runtime/`。这些文件主要供 AI 维护和审计；
运行实验不要求先阅读其实现。

## 1. 直接启动三项正式训练

在服务器 Pocket_Plus 项目根目录执行：

```bash
# Find_0：一台节点、两张 H200、当前正式基线。
bash 训练与运行/submit_task.sh \
  --sh Find_0.sh \
  --resource h200 \
  --gpus 2 \
  --cpus 24

# Find_1：一台节点、两张 H100、带五项体素监督的新基线。
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 48

# unet_c1：一台节点、一张 H100、只保留 U-Net 体素骨干。
bash 训练与运行/submit_task.sh \
  --sh unet_c1.sh \
  --resource h100 \
  --gpus 1 \
  --cpus 24
```

每条命令只调用一次 `sbatch`。默认不创建 `pre_lock`，因此作业获得资源后
立即进入第一次正式运行。若希望先占有资源、再由人工决定何时开始，添加：

```bash
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 48 \
  --hold
```

假设 Slurm 返回 Job `400001`，上述 `--hold` 会在 Job 启动后创建：

```text
/home/penghongen/Feedback/Pocket_Plus/allocations/pre_lock_400001
```

删除该文件后，第一次正式执行才开始。

## 2. 一次提交到底做了什么

以这条命令为例：

```bash
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 48 \
  --hold \
  -- train.optimizer.weight_decay=0.02
```

`submit_task.sh` 会把它展开为含义等价的 Slurm 调用：

```text
sbatch
  --job-name=Find_1
  --partition=h100
  --qos=h100g2
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=48
  --gres=gpu:h100:2
  --output=/dev/null
  --error=/dev/null
  训练与运行/sbatch/task.sbatch
  --source-root <当前项目根>
  --feedback-root /home/penghongen/Feedback/Pocket_Plus
  --task 训练与运行/sh/Find_1.sh
  --hold 1
  --resource h100
  --gpus 2
  --nodes 1
  --cpus 48
  --
  train.optimizer.weight_decay=0.02
```

这里有三个容易混淆的事实：

1. `submit_task.sh` 此时不复制项目，也不创建 release。
2. Slurm 会保存本次提交的 `task.sbatch` 内容，但 `--source-root` 仍指向可继续
   修改的项目目录。
3. Job 获得资源后，`task.sbatch` 从该项目目录加载四锁执行器；四锁执行器
   在每一次真正执行 `run_cmd` 前，才创建或复用当时最新代码的 release。

因此，排队期间修改代码会影响第一次运行；`try_lock` 期间修改代码会影响下一次
运行。它们都不要求重新申请 GPU。

关于 out 和 error 日志，它们实际输出重定向到：
```text
${feedback_root}/allocations/<jobid>/out
${feedback_root}/allocations/<jobid>/err
```
因此当前项目默认日志位置是：
```text
/home/penghongen/Feedback/Pocket_Plus/allocations/<jobid>/out
/home/penghongen/Feedback/Pocket_Plus/allocations/<jobid>/err
```

### 2.1 `--simple`：保留 Slurm 和四锁，不保存完整运行证据

不需要冻结项目代码、也不需要记录 launch 的一次性任务，可以使用：

```bash
bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
  --simple \
  --sh /home/penghongen/My_Project/Pocket_Plus/ops/materialize_filtered_stage1_preparation.sh \
  --resource cpu \
  --cpus 1
```

这条命令仍然调用一次 `sbatch`，CPU 作业仍然使用 `cpu` partition 和 `Cpu96`
QOS，四锁循环也仍然有效。差别是它不创建
`/home/penghongen/Feedback/Pocket_Plus/releases/` 和
`/home/penghongen/Feedback/Pocket_Plus/launches/` 中的任何内容。假设 Slurm
返回 Job `400002`，活动期间可以直接看到：

```text
/home/penghongen/SIMPLE_RUN/
├── pre_lock_400002                         # 仅使用 --hold 时创建
├── try_lock_400002                         # 一次执行结束后创建
├── after_lock_400002                       # 删除后结束 Job
├── kill_lock_400002                        # 仅人工要求终止当前命令时创建
├── run_cmd_400002.sh                       # 当前 allocation 下一次执行的命令
├── materialize_filtered_stage1_preparation_400002.out
└── materialize_filtered_stage1_preparation_400002.err
```

`pre_lock`、`try_lock`、`after_lock`、`kill_lock` 与 `run_cmd` 的操作含义和完整
模式相同，只是都集中到 `${HOME}/SIMPLE_RUN`，不再分成 `allocations/` 根目录
和 Job 子目录。任务最后一次执行的退出码会成为 Slurm Job 的退出码；`.out`
和 `.err` 在 Job 结束后继续保留。四个锁和 `run_cmd` 属于活动控制文件，在
Job 正常退出时清理。

`--simple` 不表示“不使用 Slurm”，也不表示“直接在登录节点运行”。它只关闭
release 和 launch 两套留证机制，适合已有独立输入、输出和幂等规则的短期任务。

## 3. release 为什么必须在每次运行前创建

下面用完整数值例子说明 Job `400001` 的时间顺序。

### 3.1 排队期间更新代码

1. 10:00 提交 Job `400001`。项目内容此时是版本 A；没有创建 release。
2. 10:20 Job 仍在排队。用户把项目同步为版本 B。
3. 11:00 Job 获得 GPU，四锁执行器即将第一次执行 `run_cmd_400001.sh`。
4. 执行器对 11:00 的发布源计算内容哈希，并创建
   `releases/Pocket_Plus_<B的哈希前12位>/Pocket_Plus/`。
5. 第一次训练从该 release 运行，因此使用版本 B，而不是提交时的版本 A。

### 3.2 `try_lock` 期间修复代码

1. 版本 B 的第一次训练报错，执行器创建 `try_lock_400001`，但保留 GPU。
2. 用户在共享项目中修复代码并同步为版本 C。
3. 用户确认 `run_cmd_400001.sh` 后，删除 `try_lock_400001`。
4. 执行器再次对当前发布源计算哈希，创建
   `releases/Pocket_Plus_<C的哈希前12位>/Pocket_Plus/`。
5. 第二次训练使用版本 C，并建立新的 launch。

若代码没有变化，内容哈希相同，下一次运行会直接复用原有 release，不重复复制。

### 3.3 一个精确边界

Job 启动时加载的 `task.sbatch` 和 `allocation_runner.sh` 是本 Job 的控制器。
在 `try_lock` 期间修改训练脚本、Python、配置和 release/launch 辅助脚本，
下一次发布会包含修改；但已经在内存中运行的 `allocation_runner.sh` 本身不会
热替换。若要更换四锁循环本身，应该结束旧 allocation 后重新提交。

这不会妨碍最常见的“修训练代码后继续使用同一批 GPU”流程。

## 4. 服务器目录：四类内容分别回答什么问题

默认反馈根为：

```text
/home/penghongen/Feedback/Pocket_Plus/
├── releases/
├── launches/
├── logs/
└── allocations/
```

这里没有额外的 `runs/`。训练产物目录只使用 `src/train.py` 已有的 `logs/`。

### 4.1 `releases/`：这一次实际从哪份代码运行

示例：

```text
releases/
└── Pocket_Plus_2a4f8d61c937/
    ├── manifest.json
    └── Pocket_Plus/
        ├── src/
        ├── configs/
        ├── 训练与运行/
        └── 与服务器交互/
```

`Pocket_Plus/` 是可执行的完整项目副本。复制时排除 `.git/`、Python 缓存和
pytest 缓存。`manifest.json` 记录：

- 完整内容 SHA-256；
- release 创建时间；
- 原始发布源；
- 项目目录名；
- 可取得时的 Git 提交、分支和工作区是否有改动。

release 的身份来自项目内容，不来自 Job ID。两个 Job 使用完全相同的项目内容时，
可以安全复用同一 release。

### 4.2 `launches/`：某一次实际执行作出了什么启动决定

同一个 Job 可在 `try_lock` 后运行多次。每次真正执行都有独立 launch：

```text
launches/
└── 400001/
    ├── Find_1_job400001_20260726T120000_a1/
    │   ├── run_cmd.sh
    │   └── launch.json
    └── Find_1_job400001_20260726T123000_a2/
        ├── run_cmd.sh
        └── launch.json
```

第一次 launch 可以指向版本 B 的 release，第二次 launch 可以指向版本 C 的
release。`launch.json` 记录该次执行的：

- release 项目根；
- 原始任务脚本相对路径；
- Job ID、资源类型、节点数、GPU 数和 CPU 数；
- Slurm array 表达式；
- 唯一 `POCKET_RUN_STAMP`。

`run_cmd.sh` 是执行前动态命令的只读副本。若人工修改过
`allocations/400001/run_cmd_400001.sh`，launch 保存的是修改后的真实命令。

release 与 launch 不重复：

- release 回答“有哪些代码和配置文件可供执行”；
- launch 回答“本 Job 的这一次尝试实际选择了哪个 release、哪条命令和哪些资源”。

### 4.3 `logs/`：训练实际产生了什么

`src/train.py` 读取最终 Hydra 配置后建立：

```text
logs/<experiment_group>/<tag>____<POCKET_RUN_STAMP>/
```

例如 Job `400001` 的 Find_1 CPC1 可能生成：

```text
logs/
└── AdaLigand_Stage1-Find_1-CPC1/
    └── Find_1-CPC1____Find_1_job400001_20260726T120000_a1_CPC1/
        ├── config.yaml
        ├── train.yaml
        ├── src_snapshot/
        ├── checkpoints/
        ├── train.log
        └── wandb/
```

这一级由训练代码管理，不由 Slurm 通用脚本重新实现：

- `config.yaml` 是最终解析配置，是判断真实训练参数的最终依据；
- `src_snapshot/` 是 `src/train.py` 保存的实际运行代码快照；
- `checkpoints/` 保存 `BEST.ckpt`、TOP、last 等模型状态；
- W&B 与训练日志保存曲线、指标和诊断。

三种证据的关系是：

```text
release：执行前的完整项目
→ launch：某次尝试选择的 release、动态命令和资源
→ logs：src/train.py 实际解析配置并运行后产生的代码快照与模型产物
```

### 4.4 `allocations/`：怎样直观控制已经申请到的资源

Job `400001` 运行时：

```text
allocations/
├── pre_lock_400001
├── try_lock_400001
└── 400001/
    ├── after_lock_400001
    ├── kill_lock_400001
    ├── run_cmd_400001.sh
    ├── out
    └── err
```

这些文件不会同时永久存在：

- `pre_lock_400001`：只有提交时使用 `--hold` 才创建。删除后开始第一次运行。
- `try_lock_400001`：一次命令成功、失败或被 kill 后创建。删除后再次运行。
- `after_lock_400001`：Job 启动即创建。删除后结束循环并释放 allocation。
- `kill_lock_400001`：默认不存在。人工创建后，哨兵终止当前 `run_cmd` 进程组，
  删除该锁，再进入 `try_lock`。
- `run_cmd_400001.sh`：下一次要执行的动态命令。只应在任务未运行且
  `pre_lock` 或 `try_lock` 存在时编辑。
- `out`、`err`：该 allocation 从开始到结束的标准输出和标准错误，不轮转。

四个常用人工动作：

```bash
# 开始 --hold 后的第一次运行
rm /home/penghongen/Feedback/Pocket_Plus/allocations/pre_lock_400001

# 终止当前训练进程，但保留 GPU，随后等待 try_lock
touch /home/penghongen/Feedback/Pocket_Plus/allocations/400001/kill_lock_400001

# 修复并同步代码、检查 run_cmd 后，开始下一次运行
rm /home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_400001

# 当前没有训练进程时，结束 Job 并释放 allocation
rm /home/penghongen/Feedback/Pocket_Plus/allocations/400001/after_lock_400001
```

不要用 `scancel` 代替正常的 `after_lock` 释放，除非明确需要 Slurm 强制终止
整个 Job。

## 5. 三份训练脚本如何参与训练

三个脚本都遵循同一阅读顺序：

```text
定位本 release
→ 激活 Conda
→ 指定两类输入数据
→ 指定训练输出根
→ 接收 allocation 的 GPU/节点数
→ 设置 Python/CUDA/线程环境
→ 选择完整 experiment
→ 用少量 Hydra 参数恢复正式基线
→ 运行 src/train.py
→ 检查 BEST.ckpt
```

它们有意重复环境和路径设置。这样单独打开任一训练脚本时，都能看到该训练
完整依赖什么，不必跳到共享 `_common.sh` 才能理解。

### 5.1 三个路径变量分别控制什么

#### `ADALIGAND_DATA_ROOT`

默认值：

```text
/storage/penghongen/AdaLigand/Ori_Data
```

Find 的完整链路：

```text
Find_0.sh 或 Find_1.sh
→ export ADALIGAND_DATA_ROOT
→ configs/dataset/stage1_find.yaml:10
→ ${oc.env:ADALIGAND_DATA_ROOT,...}
→ dataset.all_data_path
→ src/train.py: UnifiedDataModule.setup()
→ Stage1Dataset.__init__(all_data_path=...)
→ self.root = Path(all_data_path)
→ _materialize()/__getitem__()
→ 按 PDB 读取 A–G 的 density、parse、ligand、labels 和可用的 ligand_dist.npz
```

`unet_c1.sh` 经 `configs/dataset/stage1_unet_c1.yaml:9` 进入同一个
`Stage1Dataset`。它只把一个密度通道送入 U-Net，但仍读取受体原子和配体距离，
用于生成三项新增辅助标签。

改变这个变量会改变训练读取的正式 A–G 数据集。它不是训练输出位置。

#### `ADALIGAND_STAGE1_PREPARATION_ROOT`

默认值：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation
```

Find 的完整链路：

```text
训练脚本
→ export ADALIGAND_STAGE1_PREPARATION_ROOT
→ configs/dataset/stage1_find.yaml:11
→ dataset.box_pool_root=<变量>/box_pool
→ dataset.split_train=<box_pool>/train
→ dataset.split_val=<box_pool>/validation_selection.npz
→ UnifiedDataModule.setup()
→ Stage1Dataset.__init__()
→ build_request_source()
→ 训练请求或固定验证请求
→ 具体 PDB、BOX 起点、样本角色和监督开关
```

它决定“从 A–G 完整图中切哪些 80³ BOX”，不是密度与标签本身的位置。

#### `EXPERIMENT_FEEDBACK_ROOT`

默认值：

```text
/home/penghongen/Feedback/Pocket_Plus
```

完整链路：

```text
训练脚本
→ export EXPERIMENT_FEEDBACK_ROOT
→ src/train.py:53–55
→ ExperimentManager(feedback_root=...)
→ ExperimentManager._resolve_run_dir()
→ <变量>/logs/<experiment_group>/<tag>____<POCKET_RUN_STAMP>/
→ config、源码快照、checkpoint、W&B 与训练日志
```

它只控制新训练的正式输出，不改变训练输入数据。

这三个变量目前用环境变量桥接 shell 与 Hydra。未来可以迁入配置系统，但那是
独立改动；当前脚本忠实解释现有代码真实读取方式。

## 6. experiment 与 shell 覆盖怎样合并

三个脚本首先选择完整 experiment：

| 脚本 | 第一阶段或单阶段 | 第二阶段 |
| --- | --- | --- |
| `Find_0.sh` | `+experiment=CPC1/Find_0` | `+experiment=CPC2/Find_0` |
| `Find_1.sh` | `+experiment=CPC1/Find_1` | `+experiment=CPC2/Find_1` |
| `unet_c1.sh` | `+experiment=unet_c1` | 无 |

experiment 负责模型、Dataset、损失、冻结策略和常规训练制度。shell 不再建立
另一套模型/Dataset/损失入口，只显式恢复这三项正式训练曾经使用、但当前公共
YAML 可能发生变化的参数。

优先级从低到高为：

```text
base 配置
→ experiment 及其引用的 YAML
→ 训练脚本数组中的正式覆盖
→ submit_task.sh 的 `--` 之后传入的一次性 Hydra 覆盖
```

例如：

```bash
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 48 \
  -- train.optimizer.lr=1.0e-5
```

会让本次 Find_1 的 `train.optimizer.lr` 最终成为 `1e-5`，覆盖脚本中的
`5e-5`。CPC1 和 CPC2 都会收到该参数，因为两个 `python src/train.py`
命令都把 `"$@"` 放在固定覆盖之后。

最终事实始终以新运行目录中的 `config.yaml` 为准，而不是只看 YAML 或 shell。

## 7. 三项正式基线的具体数值

| 参数 | Find_0 | Find_1 | unet_c1 |
| --- | ---: | ---: | ---: |
| 推荐 GPU | H200 × 2 | H100 × 2 | H100 × 1 |
| 每卡 batch | 8 | 6 | 6 |
| 全局 batch | 48 | 48 | 48 |
| 梯度累积 | 3 | 4 | 8 |
| DataLoader workers | 10 | 10 | 20 |
| 最大学习率 | `5e-5` | `5e-5` | `1e-4` |
| CPC1/单阶段 warmup | `0.005` | `0.005` | `0.005` |
| CPC1/单阶段 patience | 2 | 3 | 3 |
| 最大 epoch | 20 | 20 | 20 |
| 每 epoch 验证次数 | 30 | 30 | 30 |
| W&B | online | online | online |

梯度累积来自：

```text
Find_0：8 BOX/卡 × 2 卡 = 16 BOX/前向；48 ÷ 16 = 累积 3 次
Find_1：6 BOX/卡 × 2 卡 = 12 BOX/前向；48 ÷ 12 = 累积 4 次
unet_c1：6 BOX/卡 × 1 卡 = 6 BOX/前向；48 ÷ 6 = 累积 8 次
```

`train.strict_global_batch_size=true` 会要求这个除法得到整数；
`train.enable_batch_size_tuning=false` 会阻止程序自动改动这些基线数值。

### 7.1 Find_0

- 选择旧版 Find_0 模型与损失，不含蛋白、核酸和配体距离辅助监督。
- 真实受体原子密度邻域边长显式恢复为 `11×11×11`。
- CPC1 最多实际降低学习率 4 次；CPC2 不 warmup，并在第一次降学习率后结束。

### 7.2 Find_1

- 选择新版 Find_1 模型、Gaussian 受体原子体素散射与五项体素监督。
- 伪原子 density cube 每块处理 1024 个 anchor。
- 真实原子 density cube 每块处理 2048 个原子，邻域边长为 `11×11×11`。
- CPC1 和本基线 CPC2 都显式使用 `warmup_ratio=0.005`。

五项 CPC1/主训练损失权重由 experiment 引用的损失 YAML 提供：

```text
ligand_area             1.00
ligand_distance         0.30
binding_area            0.10
protein_mainchain       0.05
nucleic_mainchain       0.05
```

训练脚本只解释这些最终值，不重复声明损失权重，因此修改损失仍应修改或选择
experiment 所引用的损失配置。

### 7.3 unet_c1

- 选择 density-only U-Net；没有 Find 点分支和密度—原子融合主干。
- 输出头和五项体素监督与新版 Find_1 对齐。
- 单卡批量 6、全局批量 48，因此每次优化器更新累积 8 个前向。
- 只有一个阶段，不执行 CPC2。

## 8. CPC1 怎样连接 CPC2

`Find_0.sh` 和 `Find_1.sh` 都执行：

```text
CPC1 src/train.py
→ 检查本轮 CPC1/checkpoints/BEST.ckpt
→ CPC2 init_from=<本轮 CPC1 BEST.ckpt>
→ CPC2 src/train.py
→ 检查本轮 CPC2/checkpoints/BEST.ckpt
```

如果 CPC1 报错或没有产生 `BEST.ckpt`，脚本不会启动 CPC2，并以失败状态返回
四锁执行器。allocation 随后创建 `try_lock`，允许修复代码后继续使用同一资源。

`init_from` 是模型权重初始化，不是恢复旧训练目录后继续写入；CPC1 与 CPC2
分别拥有独立的 `POCKET_RUN_STAMP` 后缀和独立 logs 目录。

## 9. 通用资源参数

| `--resource` | 默认 partition | 默认 QOS | 未写 `--cpus` 时 |
| --- | --- | --- | ---: |
| `cpu` | `cpu` | `Cpu96` | 16 |
| `a100` | `a100` | `a100g2` | 每张 GPU 16 核 |
| `a800` | `nvlink` | `nvlinkg8` | 每张 GPU 24 核 |
| `h100` | `h100` | `h100g2` | 每张 GPU 24 核 |
| `h200` | `h200` | `h200g2` | 每张 GPU 24 核 |

可用选项：

- `--nodes N`：节点数，默认 1。
- `--gpus N`：每节点 GPU 数。
- `--cpus N`：每个 Slurm task 的 CPU 核数。
- `--partition`、`--qos`：覆盖资源类型映射。
- `--nodelist NAME`：要求指定节点。
- `--mem VALUE`、`--time VALUE`：原样传给 sbatch。
- `--array SPEC`：原样传给 sbatch，例如 `0-15%4`。
- `--hold`：Job 启动后先建立可见的 `pre_lock`。

提交器不会解释 `SLURM_ARRAY_TASK_ID`。如果任务脚本没有读取这个变量，
`--array 0-15` 就会执行 16 份相同任务；数组编号的科学含义由具体任务负责。

`--sh` 只有两种解释：

1. 值是单独的 `文件名.sh`，例如 `Find_1.sh`，从 `训练与运行/sh/` 查找。
2. 其他写法不做路径转换，原样交给任务执行器。此时通常应填写服务器绝对路径。

```bash
bash 训练与运行/submit_task.sh \
  --simple \
  --sh /home/penghongen/My_Project/Pocket_Plus/ops/自定义任务.sh \
  --resource cpu \
  --cpus 16 \
  --array 0-15%4
```

第二种写法不检查脚本是否属于当前项目。完整模式仍会为当前项目建立 release 和
launch，但这个外部任务脚本本身不会复制进 release；`--simple` 模式既原样执行
该路径，也不建立 release 和 launch。路径不存在等错误由任务真正执行时的 shell
直接报告。

## 10. 覆盖发布源

默认发布源是 `submit_task.sh` 上一层，不含任何 Pocket_Plus 绝对路径。
把整个目录结构复制到另一个项目后，默认会发布那个新项目。

如确实要从同一提交入口发布另一份项目副本：

```bash
bash 训练与运行/submit_task.sh \
  --release-source ../Pocket_Plus_candidate \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2
```

相对 `--release-source` 从默认项目根解析；也可以传绝对路径。被覆盖的目录仍须
包含：

```text
训练与运行/sbatch/task.sbatch
与服务器交互/other/training_runtime/
训练与运行/sh/Find_1.sh
```

这只是一个简单覆盖入口，不要求训练脚本声明 release 来源，也没有递归保护协议。

## 11. 怎样修改实验

常见修改分三类：

1. 改模型、Dataset 或损失：修改/新建 experiment 与它引用的 YAML。
2. 改当前正式基线必须显式固定的运行参数：修改对应训练脚本中的 Hydra 数组。
3. 只做一次尝试：在提交命令的 `--` 后追加 Hydra 覆盖。

例如模型结构和损失权重不应在 shell 中再建立变量入口。训练脚本中的
`+experiment=...` 已经完整选择这些配置；shell 只覆盖明确列出的运行数值。

修改共享项目后：

- 尚未开始的排队 Job 会在第一次运行前发布新内容；
- 处于 `pre_lock` 的 Job 会在删除该锁后发布新内容；
- 处于 `try_lock` 的 Job 会在删除该锁后发布新内容；
- 正在运行的 Python 进程不会被共享项目改动影响，因为它来自既有 release。

## 12. 无卡验证边界

这套脚本可以在不申请资源时检查：

- 所有 shell 文件通过 `bash -n`；
- 用假的 `SBATCH_BIN` 查看最终 sbatch 参数；
- 验证提交阶段没有创建 release；
- 用临时任务验证第一次运行和 `try_lock` 重试分别绑定不同 release；
- 用临时 `${HOME}` 直接运行 Slurm 包装层，验证 `--simple` 的锁、动态命令、
  退出码以及“不创建 release/launch”；
- 对脚本中的 Hydra 参数做静态对照。

这些检查不代替真实 GPU smoke，但本目录的三个正式训练此前已经分别通过训练
启动验证。本次整理不提交新 Job，也不接管正在运行的 allocation。

## 13. 与既有三个 Job 的关系

Job `321107`、`321540` 和 `321743` 在本目录建立前已由旧运行体系启动。
它们继续使用各自的旧 release/runtime 和
`/home/penghongen/My_Project/feedback_plus/logs/`，不会被本目录迁移、覆盖或
接管。精确入口与运行事实保存在 AdaLigand：

```text
talk/三个训练的检查记录.md
```

以后由本目录新提交的任务才默认写入：

```text
/home/penghongen/Feedback/Pocket_Plus/
```

检查训练时，应先根据 Job 命令、launch 与 `POCKET_RUN_STAMP` 判断它属于旧体系
还是本体系，不能把新目录规则套到三个既有 Job。
