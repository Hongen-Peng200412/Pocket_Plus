# 训练与运行

本目录提供一套通用的 Slurm 任务提交方式、五项 AdaLigand Stage1 正式训练入口，以及一份本轮只组合和测试的 Find_1 PDB-centric-2 入口。

日常使用只需要理解以下三层：

```text
submit_task.sh
→ sbatch/task.sbatch
→ sh/<具体训练>.sh
```

- `submit_task.sh`：选择任务、节点数、每节点 GPU 数、CPU 数、跨节点 DDP 和可选锁。
- `sbatch/task.sbatch`：唯一真正交给 `sbatch` 的通用资源包装脚本。
- `sh/Find_0.sh`、`sh/Find_1.sh`、`sh/unet_base.sh`、`sh/unet_c1.sh`、`sh/unet_diff.sh`：正式训练入口。`Find_1.sh` 固定选择 PDB-centric-1，不根据参数切换采样制度。
- `sh/Find_1_pdb_centric_2.sh`：PDB-centric-2 的独立入口；本轮只用于配置和测试，未经新授权不提交。
- `sh/unet_c1_no_mainchain.sh`：复用 `unet_c1.sh` 的辅助损失消融薄包装。

release、launch 与四种锁控制的实现位于
`训练与运行/runtime/`。这些文件主要供 AI 维护和审计；
运行实验不要求先阅读其实现。

跨节点训练额外经过 `runtime/launch_training_python.sh`。allocation 控制器在每个节点启动一个 `torchrun` agent，agent 再为本节点的每张 GPU 创建一个 Python 训练进程；单节点训练仍直接运行一个 Python 主进程并由 Lightning 启动本节点的 DDP 子进程。

本文中的“任务根目录”是一次提交需要冻结的完整项目目录。未填写
`--task-root` 时，它就是 `训练与运行` 的上一层；因此把整个 `训练与运行`
复制到 AdaLigand 等另一个项目后，提交器会默认冻结那个项目，不依赖
`Pocket_Plus` 这个名称。

## 1. 直接启动五项正式训练

在服务器 Pocket_Plus 项目根目录执行：

```bash
# Find_0：一台节点、两张 H100，CPC1 训练。
bash 训练与运行/submit_task.sh \
  --sh Find_0.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 32

# Find_1 PDB-centric-1：一台节点、两张 H100，从头训练。
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 64

# Find_1 PDB-centric-1：一台节点、两张 H100，从头训练。
bash 训练与运行/submit_task.sh \
  --sh Find_1_pdb_centric_2.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 64

# unet_base：一台节点、一张 A800。
bash 训练与运行/submit_task.sh \
  --sh unet_base.sh \
  --resource a800 \
  --gpus 1 \
  --cpus 16

# unet_c1 采样方式三主链版：一台节点、一张 H100；退出后保留 allocation。
bash 训练与运行/submit_task.sh \
  --sh unet_c1.sh \
  --resource h100 \
  --gpus 1 \
  --cpus 32 \
  --after_hold

# unet_diff：一台节点、一张 A800。
bash 训练与运行/submit_task.sh \
  --sh unet_diff.sh \
  --resource a800 \
  --gpus 1 \
  --cpus 16
```

`unet_c1_no_mainchain.sh` 复用 `unet_c1.sh` 并把 protein/nucleic 辅助损失权重设为 0；双卡提交时使用 32 CPU。

历史 Find_1 续训不使用当前生产 `sh/Find_1.sh`。当前脚本固定执行 PDB-centric-1 从头训练；历史续训只存在于从提交 `1315d301c867c99e2dc0736feffde86e9cd7fa0a` 建立的隔离分支 `codex/find1-historical-resume`。正式提交必须使用该分支生成的独立 release，并以 [Find_1 训练与历史续训执行记录](../文档/exec_plan/2026-08-31_Find_1训练与历史续训.md) 中冻结的命令、checkpoint 和动态 CPU 参数为准。不得从当前生产任务根把 `Find_1.sh` 当作历史续训入口。

每条命令只调用一次 `sbatch`。默认不创建 `pre_lock` 或 `try_lock`：作业获得资源后
立即执行一次任务，任务结束后自动退出并释放资源。若希望先占有资源、再由人工决定何时开始，添加：

```bash
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 64 \
  --pre_hold
```

假设 Slurm 返回 Job `400001`，上述 `--pre_hold` 会在 Job 启动后创建：

```text
/home/penghongen/Feedback/Pocket_Plus/allocations/pre_lock_400001
```

删除该文件后，第一次正式执行才开始。

若希望任务结束后继续保留 allocation 和计算资源，添加 `--after_hold`。执行器会在每次
任务结束后创建 `try_lock`；删除 `try_lock` 会再次执行当前动态命令，删除 `after_lock`
才会结束 Job 并释放资源。`--pre_hold` 与 `--after_hold` 相互独立，可以只用一个或同时使用。

## 2. 一次提交到底做了什么

以这条命令为例：

```bash
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 64 \
  --pre_hold \
  --after_hold \
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
  --cpus-per-task=64
  --gres=gpu:h100:2
  --output=/dev/null
  --error=/dev/null
  训练与运行/sbatch/task.sbatch
  --task-root <当前项目根>
  --feedback-root /home/penghongen/Feedback/Pocket_Plus
  --task 训练与运行/sh/Find_1.sh
  --pre_hold 1
  --after_hold 1
  --resource h100
  --gpus 2
  --nodes 1
  --multi_node_ddp 0
  --cpus 64
  --
  train.optimizer.weight_decay=0.02
```

这里有三个容易混淆的事实：

1. `submit_task.sh` 此时不复制项目，也不创建 release。
2. Slurm 会保存本次提交的 `task.sbatch` 内容，但 `--task-root` 仍指向可继续
   修改的项目目录。
3. Job 获得资源后，`task.sbatch` 从该项目目录加载 allocation 执行器；执行器
   在每一次真正执行 `run_cmd` 前，才创建或复用当时最新代码的 release。

因此，排队期间修改代码会影响第一次运行；启用 `--after_hold` 后，`try_lock` 期间
修改代码会影响下一次运行。它们都不要求重新申请 GPU。

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

### 2.1 `--simple`：保留 Slurm 和锁控制，不保存完整运行证据

不需要冻结项目代码、也不需要记录 launch 的一次性任务，可以使用：

```bash
bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
  --simple \
  --task-root /home/penghongen/My_Project/Pocket_Plus \
  --sh /home/penghongen/My_Project/Pocket_Plus/ops/stage1_data_preparation/run/finalize_box_pool_3.sh \
  --resource cpu \
  --cpus 1
```

这条命令仍然调用一次 `sbatch`，CPU 作业仍然使用 `cpu` partition 和 `Cpu96`
QOS，锁控制也仍然有效。差别是它不创建
`/home/penghongen/Feedback/Pocket_Plus/releases/` 和
`/home/penghongen/Feedback/Pocket_Plus/launches/` 中的任何内容。假设 Slurm
返回 Job `400002`，活动期间可以直接看到：

```text
/home/penghongen/SIMPLE_RUN/
├── pre_lock_400002                         # 仅使用 --pre_hold 时创建
├── try_lock_400002                         # 仅使用 --after_hold 时在执行结束后创建
├── after_lock_400002                       # 删除后结束 Job
├── kill_lock_400002                        # 仅人工要求终止当前命令时创建
├── run_cmd_400002.sh                       # 当前 allocation 下一次执行的命令
├── finalize_box_pool_3_400002.out
└── finalize_box_pool_3_400002.err
```

`pre_lock`、`try_lock`、`after_lock`、`kill_lock` 与 `run_cmd` 的操作含义和完整
模式相同，只是都集中到 `${HOME}/SIMPLE_RUN`，不再分成 `allocations/` 根目录
和 Job 子目录。任务最后一次执行的退出码会成为 Slurm Job 的退出码；`.out`
和 `.err` 在 Job 结束后继续保留。四个锁和 `run_cmd` 属于活动控制文件，在
Job 正常退出时清理。

`--simple` 不表示“不使用 Slurm”，也不表示“直接在登录节点运行”。它只关闭
release 和 launch 两套留证机制，适合已有独立输入、输出和幂等规则的短期任务。
任务脚本仍须位于任务根目录内；这保证完整模式与 simple 模式使用同一套路径
解释，不会出现某个绝对路径被悄悄当作特殊“外部任务”的情况。

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

1. 提交时启用了 `--after_hold`；版本 B 的第一次训练报错后，执行器创建 `try_lock_400001` 并保留 GPU。
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
- 是否启用跨节点 DDP、实际启动器、Slurm 节点表达式、rendezvous 主节点和端口；
- Slurm array 表达式；
- 唯一 `TASK_RUN_STAMP`，即本次实际执行传给训练程序的目录标识。

`run_cmd.sh` 是执行前动态命令的只读副本。若人工修改过
`allocations/400001/run_cmd_400001.sh`，launch 保存的是修改后的真实命令。

release 与 launch 不重复：

- release 回答“有哪些代码和配置文件可供执行”；
- launch 回答“本 Job 的这一次尝试实际选择了哪个 release、哪条命令和哪些资源”。

### 4.3 `logs/`：训练实际产生了什么

`src/train.py` 读取最终 Hydra 配置后建立：

```text
logs/<experiment_group>/<tag>____<TASK_RUN_STAMP>/
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

- `pre_lock_400001`：只有提交时使用 `--pre_hold` 才创建。删除后开始第一次运行。
- `try_lock_400001`：只有提交时使用 `--after_hold`，才在一次命令成功、失败或被 kill 后创建。删除后再次运行。
- `after_lock_400001`：Job 启动即创建。删除后结束循环并释放 allocation。
- `kill_lock_400001`：默认不存在。人工创建后，哨兵终止当前 `run_cmd` 进程组，
  删除该锁，再进入 `try_lock`。
- `run_cmd_400001.sh`：下一次要执行的动态命令。只应在任务未运行且
  `pre_lock` 或 `try_lock` 存在时编辑。
- `out`、`err`：该 allocation 从开始到结束的标准输出和标准错误，不轮转。

四个常用人工动作：

```bash
# 开始 --pre_hold 后的第一次运行
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

## 5. 训练脚本如何参与训练

当前训练脚本都遵循同一阅读顺序：

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

Find、unet_base 与 unet_diff 的 PDB 中心采样链路：

```text
Find_0.sh、Find_1.sh 或 Find_1_pdb_centric_2.sh
→ export ADALIGAND_DATA_ROOT
→ 对应的 configs/dataset/stage1_find*.yaml
→ ${oc.env:ADALIGAND_DATA_ROOT,...}
→ dataset.all_data_path
→ src/train.py: UnifiedDataModule.setup()
→ Stage1Dataset.__init__(all_data_path=...)
→ self.root = Path(all_data_path)
→ _materialize()/__getitem__()
→ 按 PDB 读取 A–G 的 density、parse、ligand、labels 和可用的 ligand_dist.npz
```

`unet_base.sh`、`unet_c1.sh` 与 `unet_diff.sh` 分别经同名 Dataset 配置进入同一个 `Stage1Dataset`。三者只改变送入 U-Net 的密度通道，仍读取受体原子和配体距离以生成三项辅助标签。

改变这个变量会改变训练读取的正式 A–G 数据集。它不是训练输出位置。

#### `ADALIGAND_STAGE1_PREPARATION_ROOT`

默认值：

```text
/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3
```

Find 的完整链路：

```text
训练脚本
→ export ADALIGAND_STAGE1_PREPARATION_ROOT
→ 对应的 Dataset 配置
→ dataset.box_pool_root=<变量>/box_pool
→ dataset.split_train=<box_pool>/train
→ dataset.split_val=<box_pool>/对应的 PDB-centric-1/V1 或 PDB-centric-2/V2 冻结文件
→ UnifiedDataModule.setup()
→ Stage1Dataset.__init__()
→ build_request_source()
→ PDB-centric-1/V1 或 PDB-centric-2/V2 请求
→ 具体 PDB、BOX 起点、样本角色和监督开关
```

`Find_1_pdb_centric_2.sh` 与 `unet_c1.sh` 使用同一 BOX pool，并各自通过 Dataset 配置指向 V2 验证文件：

```text
configs/dataset/stage1_find_pdb_centric_2.yaml
或 configs/dataset/stage1_unet_c1.yaml
→ dataset.split_train=<box_pool>/train
→ dataset.split_val=<box_pool>/validation_selection_pdb_centric_v2.npz
→ PDB-centric-2 的动态训练请求或 V2 固定验证请求
```

这些路径决定“从 A–G 完整图中切哪些 80³ BOX”，不是密度与标签本身的位置。方式二固定 `25/0.5/25`，每个 PDB 使用 25 个 bias 与 25 个 context；V1 以 seed 3407 从 200 个 validation PDB 中冻结 150 个身份。方式三固定 `50/0.7575757575757576/1`，每个 occurrence 最多贡献一个 bias，并为每个 PDB 抽取 16 个 context；V2 冻结全部 200 个 validation PDB。

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
→ <变量>/logs/<experiment_group>/<tag>____<TASK_RUN_STAMP>/
→ config、源码快照、checkpoint、W&B 与训练日志
```

它只控制新训练的正式输出，不改变训练输入数据。

这三个变量目前用环境变量桥接 shell 与 Hydra。未来可以迁入配置系统，但那是
独立改动；当前脚本忠实解释现有代码真实读取方式。

## 6. experiment 与 shell 覆盖怎样合并

训练脚本首先选择完整 experiment：

| 脚本 | 当前活动 experiment | 后续阶段 |
| --- | --- | --- |
| `Find_0.sh` | `+experiment=CPC1/Find_0` | 不自动启动 CPC2 |
| `Find_1.sh` | `+experiment=CPC1/Find_1` | 不自动启动 CPC2 |
| `Find_1_pdb_centric_2.sh` | `+experiment=CPC1/Find_1_pdb_centric_2` | 本轮不提交训练 |
| `unet_base.sh` | `+experiment=unet_base` | 无 |
| `unet_c1.sh` | `+experiment=unet_c1` | 无 |
| `unet_diff.sh` | `+experiment=unet_diff` | 无 |

experiment 负责模型、Dataset、损失、冻结策略和训练制度。Find_1 的两份入口不含采样条件分支；PDB-centric-1 与 PDB-centric-2 分别选择独立的 Dataset、train 和 experiment YAML。shell 只补充设备数、节点数、运行身份和少量显存相关参数。

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
  --cpus 64 \
  -- train.optimizer.lr=1.0e-5
```

会让本次 Find_1 的 `train.optimizer.lr` 最终成为 `1e-5`，覆盖 PDB-centric-1 train YAML 中的 `5e-5`。当前 Find 脚本只启动 CPC1，并把 `"$@"` 放在固定覆盖之后。

最终事实始终以新运行目录中的 `config.yaml` 为准，而不是只看 YAML 或 shell。

## 7. 五项正式基线的具体数值

| 参数 | Find_0 | Find_1 | unet_base | unet_c1 | unet_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| 推荐 GPU | H100 × 2 | H100 × 2 | A800 × 1 | H100 × 1 | A800 × 1 |
| Slurm CPU | 32 | 64 | 16 | 32 | 16 |
| 每卡 batch | 8 | 6 | 8 | 8 | 8 |
| 全局 batch | 48 | 48 | 48 | 48 | 48 |
| 梯度累积 | 3 | 4 | 6 | 6 | 6 |
| DataLoader workers | 每 rank 16，总 32 | 每 rank 24，总 48 | 16 | 30 | 16 |
| 最大学习率 | `5e-5` | `5e-5` | `1e-4` | `1e-4` | `1e-4` |
| CPC1/单阶段 warmup | `0.005` | `0.005` | `0.005` | `0.005` | `0.005` |
| CPC1/单阶段 patience | 2 | 3 | 3 | 3 | 3 |
| 最大 epoch | 70 | 70 | 70 | 110 | 70 |
| 每 epoch 验证次数 | 10 | 12 | 12 | 8 | 12 |
| W&B | online | online | online | online | online |

梯度累积来自：

```text
Find_0：8 BOX/卡 × 2 卡 = 16 BOX/前向；48 ÷ 16 = 累积 3 次
Find_1：6 BOX/卡 × 2 卡 = 12 BOX/前向；48 ÷ 12 = 累积 4 次
unet_base/unet_c1/unet_diff：8 BOX/卡 × 1 卡 = 8 BOX/前向；48 ÷ 8 = 累积 6 次
```

`train.strict_global_batch_size=true` 会要求这个除法得到整数；
`train.enable_batch_size_tuning=false` 会阻止程序自动改动这些基线数值。

Find_0、Find_1 PDB-centric-1、unet_base 与 unet_diff 使用采样方式二：每个 epoch 固定包含 685,850 个训练 BOX，V1 验证固定为 150 个 PDB、7,500 个 BOX。Find_0 的 10 次 validation 对应相邻验证事件之间约 68,585 个训练 BOX，其余方式二入口的 12 次对应约 57,154 个。

`unet_c1` 使用采样方式三：每个 epoch 包含 216,739 个 bias 和 219,472 个 context，共 436,211 个训练 BOX；`val_per_epoch=8` 对应相邻验证事件之间约 54,526 个训练 BOX。V2 验证固定全部 200 个 validation PDB，包含 3,305 个 bias 和 3,200 个 context，共 6,505 个 BOX。`max_epochs=110` 与 `warmup_ratio=0.005` 使 warmup 期间经过的 BOX 数量与方式二近似一致。

### 7.1 Find_0

- 选择旧版 Find_0 模型与损失，不含蛋白、核酸和配体距离辅助监督。
- 真实受体原子密度邻域边长显式恢复为 `11×11×11`。
- CPC1 最多实际降低学习率 4 次；CPC2 不 warmup，并在第一次降学习率后结束。

### 7.2 Find_1

- 选择新版 Find_1 模型、Gaussian 受体原子体素散射与五项体素监督。
- PDB-centric-1 使用 `25/0.5/25`、150 个 validation PDB、物理 batch 6、全局 batch 48、70 个 epoch 和每 epoch 12 次 validation。
- PDB-centric-2 使用 `50/(25/33)/1`、全部 200 个 validation PDB、物理 batch 6、全局 batch 48、110 个 epoch 和每 epoch 8 次 validation；本轮只配置与测试。
- 两份新训练配置都使用一个 AdamW。`embed_head` 的体素输入投影、带 offset 的体素输出投影和整个 voxel backbone 组成体素组，其余可训练参数组成另一组；两个组各自按范数 0.5 裁剪。历史 checkpoint 续训仍保留原来的单次全局 0.5 裁剪。
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

### 7.3 三项 U-Net

- `unet_base` 使用完整 56 通道密度输入，`unet_c1` 只使用实验密度，`unet_diff` 使用实验密度与差分密度。
- 三者都选择 density-only U-Net；没有 Find 点分支和密度—原子融合主干。
- 输出头和五项体素监督与新版 Find_1 对齐。
- 单卡批量 8、全局批量 48，因此每次优化器更新累积 6 个前向。
- 只有一个阶段，不执行 CPC2。
- `unet_base` 与 `unet_diff` 保持采样方式二；本轮 `unet_c1` 独立使用采样方式三和 V2 验证文件。三种 `unet_c1` 采样实验最终仍须在同一 complete-map 数据与指标契约下比较，不直接以各自 BOX validation 分数决定胜负。

## 8. CPC1 与 CPC2 的执行边界

当前 `Find_0.sh` 和 `Find_1.sh` 只执行 CPC1，并在退出前检查各自本轮正式运行目录中的 `checkpoints/BEST.ckpt`。CPC1 报错或没有产生 `BEST.ckpt` 时，脚本以失败状态返回 allocation 执行器。

CPC2 配置能力继续保留，但这两个正式入口不会自动串联 CPC2。未来若启动 CPC2，必须由独立任务显式选择 CPC2 experiment 并设置 `init_from=<CPC1 BEST.ckpt>`；`init_from` 表示模型权重初始化，不是继续写入原 CPC1 训练目录。

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
- `--multi-node-ddp`：显式启用跨节点 DDP；只允许 GPU 任务，并要求 `--nodes` 大于 1。未提供该开关时，提交器拒绝 `--nodes` 大于 1，避免额外节点被申请后闲置。
- `--gpus N`：每节点 GPU 数。
- `--cpus N`：每个 Slurm task 的 CPU 核数。
- `--partition`、`--qos`：覆盖资源类型映射。
- `--nodelist NAME`：要求指定节点。
- `--mem VALUE`、`--time VALUE`：原样传给 sbatch。
- `--array SPEC`：原样传给 sbatch，例如 `0-15`。
- `--pre_hold`：Job 启动后先建立可见的 `pre_lock`，删除后才开始第一次执行。
- `--after_hold`：任务结束后建立 `try_lock` 并保留 allocation；默认不保留。

提交器不会解释 `SLURM_ARRAY_TASK_ID`。如果任务脚本没有读取这个变量，
`--array 0-15` 就会执行 16 份相同任务；数组编号的科学含义由具体任务负责。

`--sh` 接受三种等价写法：

1. 单独的文件名，例如 `Find_1.sh`，从 `<任务根目录>/训练与运行/sh/` 查找；
2. 从任务根目录开始的相对路径，例如 `训练与运行/sh/Find_1.sh`；
3. 指向同一文件的绝对路径，例如
   `/home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/Find_1.sh`。

提交器会把三种写法都解析成真实文件位置，再保存相对于任务根目录的路径。因此
它们进入 release 后执行的是同一个脚本。若脚本不在任务根目录内，提交器会在
调用 `sbatch` 前报错；此时应使用 `--task-root` 选择包含该脚本的项目。

```bash
bash 训练与运行/submit_task.sh \
  --simple \
  --task-root /home/penghongen/My_Project/Pocket_Plus \
  --sh /home/penghongen/My_Project/Pocket_Plus/ops/自定义任务.sh \
  --resource cpu \
  --cpus 16 \
  --array 0-15
```

这个 simple 示例不会创建 release 和 launch，但仍使用同一任务根目录解释脚本
位置。若将同一脚本改为完整模式，它会随整个项目进入 release，不存在另外一套
“外部脚本直接执行”的行为。

## 10. 覆盖发布源

本节标题沿用原有章节范式；当前命令行参数把“发布源”统一称为“任务根目录”。
默认任务根目录是 `submit_task.sh` 所在 `训练与运行` 的上一层，不含任何
Pocket_Plus 绝对路径。把整个 `训练与运行` 复制到另一个项目后，默认会冻结
那个项目。

如需从当前提交入口运行另一个项目：

```bash
bash 训练与运行/submit_task.sh \
  --task-root /home/penghongen/My_Project/AdaLigand \
  --sh /home/penghongen/My_Project/AdaLigand/训练与运行/sh/train.sh \
  --resource h100 \
  --gpus 1
```

相对 `--task-root` 从默认任务根目录解析；也可以传绝对路径。所选目录应包含：

```text
训练与运行/
├── submit_task.sh
├── README.md
├── sbatch/task.sbatch
├── runtime/
│   ├── allocation_runner.sh
│   ├── create_launch.sh
│   └── create_release.sh
└── sh/<项目自己的任务脚本>.sh
```

`训练与运行/README.md` 和 `训练与运行/sh/` 是人类使用入口。复制到新项目时，
应把示例任务替换成新项目自己的任务脚本，并相应更新 README；`submit_task.sh`、
`sbatch/` 与 `runtime/` 可以保持不变。

## 11. 怎样修改实验

常见修改分三类：

1. 改模型、Dataset 或损失：修改/新建 experiment 与它引用的 YAML。
2. 改当前正式基线必须显式固定的运行参数：修改对应训练脚本中的 Hydra 数组。
3. 只做一次尝试：在提交命令的 `--` 后追加 Hydra 覆盖。

例如模型结构和损失权重不应在 shell 中再建立变量入口。训练脚本中的
`+experiment=...` 已经完整选择这些配置；shell 只覆盖明确列出的运行数值。

修改共享项目后：

- 尚未开始的排队 Job 会在第一次运行前发布新内容；
- 使用 `--pre_hold` 且处于 `pre_lock` 的 Job 会在删除该锁后发布新内容；
- 使用 `--after_hold` 且处于 `try_lock` 的 Job 会在删除该锁后发布新内容；
- 正在运行的 Python 进程不会被共享项目改动影响，因为它来自既有 release。

## 12. 无卡验证边界

这套脚本可以在不申请资源时检查：

- 所有 shell 文件通过 `bash -n`；
- 用假的 `SBATCH_BIN` 查看最终 sbatch 参数；
- 验证提交阶段没有创建 release；
- 用临时任务验证第一次运行和 `try_lock` 重试分别绑定不同 release；
- 用临时 `${HOME}` 直接运行 Slurm 包装层，验证 `--simple` 的锁、动态命令、
  退出码以及“不创建 release/launch”；
- 用假的 `scontrol` 与 `srun` 验证一个 attempt 只创建一次 release/launch，并把同一命令分发到两个节点；
- 验证 `kill_lock` 先向活动 `srun` job step 发送 TERM，再回到既有 `try_lock` 流程；
- 精确核对 `torchrun` 的节点数、每节点进程数、节点 rank、主节点地址和端口，并让四个本地 Gloo rank 完成一次 all-reduce；
- 对脚本中的 Hydra 参数做静态对照。

这些检查不代替服务器真实双节点 NCCL smoke。没有执行记录支持时，不得把静态检查表述成已经完成的单节点训练启动验证。跨节点代码进入服务器正式训练前，还应使用两个节点执行短 smoke，核对每个 rank 的 hostname、全局 rank、NCCL 初始化、一次优化器更新、rank 0 checkpoint 和 `kill_lock` 清理；对应 Job、命令、release、launch、日志和产物路径必须写入执行记录。

## 13. 与既有提交系统的关系

`训练与运行` 是当前正式提交入口。项目中旧的 `与服务器交互/sbatch/` 只保留
历史模板或特定兼容用途，不再作为新任务的默认入口；其中的脚本可能固定旧环境、
旧路径或旧训练命令，不能仅因文件名相似就替代本目录。

已经启动的 Slurm Job 会继续使用提交时保存的 `task.sbatch` 和 Job 启动时加载的
allocation 控制器。移动当前仓库中的 runtime 不会热替换这些正在运行的控制器，
也不会覆盖既有 release、launch、训练日志或 checkpoint。

判断一次新任务属于本体系时，依次核对：

1. Slurm 命令中的 `训练与运行/sbatch/task.sbatch`；
2. launch 中的 `release_project_root`、`task_script` 与 `task_run_stamp`；
3. release 的 `manifest.json`；
4. 训练目录中的最终 `config.yaml`、`src_snapshot/` 与 checkpoint。

单次 Job 编号、某次运行的实际目录和事故处理过程属于运行记录，不写入本 README。
