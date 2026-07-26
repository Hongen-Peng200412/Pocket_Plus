#!/usr/bin/env bash

# 任一命令失败立即退出；读取未定义变量时报错；管道中任一命令失败即判失败。
# 这避免训练缺少数据或配置时仍继续写出看似正常的产物。
set -euo pipefail

# 本文件只描述 Find_1 的正式训练。它不申请 GPU、不创建 release、不实现锁，
# 也不运行 smoke test。通常由 ../submit_task.sh 申请两张 H100 后执行。
#
# 训练分两阶段：
# 1. CPC1 从头训练；
# 2. CPC1 正常结束并产生 BEST.ckpt 后，CPC2 从该 checkpoint 初始化。

# BASH_SOURCE[0] 是当前脚本文件。先得到“训练与运行/sh”的绝对位置，
# 使后续路径不依赖用户从哪个目录执行命令。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# 当前脚本位于“项目根/训练与运行/sh”，向上两级就是本次 release 的项目根。
# 因此 src/train.py、configs/ 和源码依赖全部来自同一份 release。
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

# 项目目录名用于构造默认反馈位置；标准 release 内该名称为 Pocket_Plus。
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"

# Conda 安装根目录。若服务器通过 CONDA_BASE 显式指定其他位置，则优先使用；
# 否则读取当前用户主目录下的 anaconda3。
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"

# 训练环境名。POCKET_CONDA_ENV 可在提交前覆盖，默认使用项目已验证的 CUDA 环境。
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

# A–G 正式产物根目录。该值经以下路径参与训练：
# 本环境变量
# → configs/dataset/stage1_find.yaml:10 的 ${oc.env:ADALIGAND_DATA_ROOT,...}
# → Hydra 字段 dataset.all_data_path
# → src/train.py 的 UnifiedDataModule.setup()（约第 968–988 行）
# → Stage1Dataset.__init__(all_data_path=...)（stage1_dataset.py:410–440）
# → Stage1Dataset._materialize() / __getitem__() 按 PDB 读取 density、parse、
#   ligand、labels 与 ligand_dist.npz，并生成一个 80×80×80 BOX。
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"

# Stage1 请求准备产物根目录。该值经以下路径参与训练：
# 本环境变量
# → configs/dataset/stage1_find.yaml:11
# → dataset.box_pool_root=<本目录>/box_pool
# → dataset.split_train=<box_pool>/train
# → dataset.split_val=<box_pool>/validation_selection.npz
# → UnifiedDataModule.setup() 把 split_train/split_val 分别传给训练与验证 Dataset
# → Stage1Dataset.__init__() 调用 build_request_source()（stage1_dataset.py:470–477）
# → 请求层决定每个 epoch 使用哪些 PDB、BOX 起点和 1:5:3 角色。
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000}"

# 正式训练产物根目录。src/train.py:53–55 读取该环境变量并传给
# ExperimentManager；ExperimentManager._resolve_run_dir()
#（src/utils/experiment_manager.py:97–104）最终建立：
# <本目录>/logs/<experiment_group>/<tag>____<POCKET_RUN_STAMP>/。
# config.yaml、源码快照、checkpoint、W&B 与 train.log 都写在该运行目录中。
export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"

# allocation_runner.sh 会把 submit_task.sh 申请的“每节点 GPU 数”写入 TASK_GPUS。
# 直接运行本脚本时没有该变量，因此回退到 Find_1 基线的两张 GPU。
devices="${TASK_GPUS:-2}"

# allocation_runner.sh 同样传入节点数；当前正式基线是一台节点。
nnodes="${TASK_NNODES:-1}"

# train.devices 必须是正整数，否则 Lightning 无法确定每节点启动多少训练进程。
[[ "${devices}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[Find_1][错误] TASK_GPUS 必须是正整数。" >&2
    exit 2
}

# train.nnodes 必须是正整数，否则全局批量与分布式进程数无法正确计算。
[[ "${nnodes}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[Find_1][错误] TASK_NNODES 必须是正整数。" >&2
    exit 2
}

# conda.sh 在部分版本中会读取尚未定义的内部变量，所以激活环境时暂时关闭
# nounset；环境激活完成后立即恢复 set -u。
set +u
# 加载 conda shell 函数，后续 conda activate 才能在当前 shell 生效。
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# 激活包含 PyTorch、Lightning、Hydra 与本项目依赖的正式训练环境。
conda activate "${CONDA_ENV_NAME}"
# 恢复“使用未定义变量即报错”的保护。
set -u

# 把本 release 的项目根放到 Python 模块搜索路径首位，确保 import src...
# 读取当前 release，而不是服务器上另一份同名项目。
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Hydra 报错时输出完整异常链，数据或配置错误不会只留下折叠后的短消息。
export HYDRA_FULL_ERROR=1

# Python 标准输出不缓存，使训练进度和 traceback 立即进入 allocation 的 out/err。
export PYTHONUNBUFFERED=1

# 每个 DataLoader/训练进程只给 OpenMP 一个计算线程，避免多进程乘法式抢占 CPU。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# MKL 同样限制为一个线程，避免 NumPy/PyTorch CPU 算子在每个 worker 内再并行。
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# OpenBLAS 同样限制为一个线程，使 --cpus 分配主要服务于 DataLoader workers。
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# 限制 PyTorch CUDA caching allocator 的大块切分方式，降低 80³ 训练中的显存碎片。
# 外部显式设置 PYTORCH_CUDA_ALLOC_CONF 时保留外部值。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

# 本脚本由一个 Slurm task 内的 Lightning 自行创建训练进程。清除父 Job 的 task
# 编号，避免 Lightning/torch.distributed 把 allocation 元数据误当成已启动的
# 多进程 worker 身份；GPU 可见性等 Slurm 变量不删除。
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID

# src/train.py 与 Hydra 的 configs/ 都按项目根相对路径工作，因此切换到本 release。
cd "${PROJECT_ROOT}"

# 共同覆盖同时用于 CPC1 与 CPC2。数组中的每项都是 Hydra 命令行参数；
# 它们覆盖 experiment YAML，但仍可被本脚本最后的 "$@" 再次覆盖。
common_overrides=(
    "project_name=AdaLigand_Stage1"                         # W&B 项目名；不改变本地 experiment_group 目录。
    "train.devices=${devices}"                              # 每节点训练进程/GPU 数，通常为 2。
    "train.nnodes=${nnodes}"                                # 参与训练的节点数，当前通常为 1。
    "train.ddp_find_unused_parameters=true"                 # Find 含条件分支，DDP 允许某轮存在未用参数。
    "train.global_batch_size=48"                            # 每次优化器更新等价处理 48 个 BOX。
    "train.batch_size=6"                                    # 每张 GPU 每次前向处理 6 个 BOX。
    "train.strict_global_batch_size=true"                   # 要求全局批量可由设备数与单卡批量严格整除。
    "train.enable_batch_size_tuning=false"                  # 禁止运行时自动改变已核定的单卡批量。
    "train.num_workers=10"                                  # 每个训练 DataLoader 使用 10 个读取进程。
    "train.max_epochs=20"                                   # CPC1/CPC2 各自最多运行 20 个 epoch。
    "train.val_per_epoch=30"                                # 每个 epoch 等间隔执行 30 次完整验证。
    "train.optimizer.lr=5.0e-5"                             # Find_1 的最大学习率。
    "model.backbone.density_cube_cfg.chunk_size=1024"       # 伪原子密度 cube 每块最多处理 1024 个 anchor。
    "model.backbone.real_density_cube_cfg.chunk_size=2048"  # 真实受体原子密度 cube 每块最多处理 2048 个原子。
    "model.backbone.real_atom_density_cube_size=11"         # 每个真实原子截取 11×11×11 的密度邻域。
    "model.backbone.real_density_cube_cfg.cube_size=11"     # 真实原子 cube 编码器采用同一边长 11。
    "offline=false"                                         # 优先在线记录 W&B；网络异常由 train.py 现有逻辑处理。
)

# CPC1 选择完整 Find_1/CPC1 experiment，从头训练，并恢复正式基线的调度器参数。
cpc1_overrides=(
    "+experiment=CPC1/Find_1"                    # 选择 Find_1 模型、Stage1 Dataset、五项损失与 CPC1 冻结策略。
    "init_from=null"                              # 不加载旧 checkpoint，CPC1 从头初始化。
    "${common_overrides[@]}"                      # 应用上方两阶段共同的正式覆盖。
    "train.scheduler.warmup_ratio=0.005"          # 用估算总步数的前 0.5% 线性升到 5e-5。
    "train.scheduler.patience=3"                  # 连续第 4 次验证无足够改进时才降低学习率。
    "train.scheduler.stop_after_lr_reductions=4"  # 实际发生第 4 次降学习率后结束 CPC1。
)

# allocation_runner.sh 通常已提供全局唯一 POCKET_RUN_STAMP；直接运行时用时间生成。
run_stamp_base="${POCKET_RUN_STAMP:-Find_1_$(date '+%Y%m%dT%H%M%S')}"

# CPC1 与 CPC2 在同一训练链中使用不同后缀，因而不会写入同一运行目录。
cpc1_stamp="${run_stamp_base}_CPC1"
cpc2_stamp="${run_stamp_base}_CPC2"

# 这里按 ExperimentManager._resolve_run_dir() 的真实规则预先计算 CPC1 目录，
# 只用于碰撞检查与定位本轮 BEST.ckpt。
cpc1_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____${cpc1_stamp}"

# 同理预先计算 CPC2 目录；CPC2 只有在 CPC1 成功后才会真正创建。
cpc2_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/AdaLigand_Stage1-Find_1-CPC2/Find_1-CPC2____${cpc2_stamp}"

# 禁止覆盖同名 CPC1 运行；碰撞通常表示重复使用了 POCKET_RUN_STAMP。
[[ ! -e "${cpc1_run}" ]] || {
    echo "[Find_1][错误] CPC1 目录已存在：${cpc1_run}" >&2
    exit 23
}

# 同时提前保护 CPC2，避免 CPC1 完成后才发现目标目录已存在。
[[ ! -e "${cpc2_run}" ]] || {
    echo "[Find_1][错误] CPC2 目录已存在：${cpc2_run}" >&2
    exit 23
}

# 让 src/train.py 用 CPC1 后缀建立唯一运行目录。
export POCKET_RUN_STAMP="${cpc1_stamp}"

# 把即将写入的 CPC1 目录打印到 allocation 的 out，便于人工定位。
echo "[Find_1] 启动 CPC1：${cpc1_run}"

# 启动正式 CPC1。"$@" 位于数组之后，因此提交命令中 `--` 后的 Hydra 参数
# 拥有最高优先级，可用于一次实验覆盖而无需修改本文件。
python -u src/train.py "${cpc1_overrides[@]}" "$@"

# CPC2 只接受本轮 CPC1 由 ModelCheckpoint 保存的 BEST.ckpt。
cpc1_best="${cpc1_run}/checkpoints/BEST.ckpt"

# CPC1 没有 BEST.ckpt 说明训练未完成正式验收，不允许静默进入 CPC2。
[[ -f "${cpc1_best}" ]] || {
    echo "[Find_1][错误] CPC1 没有产生 BEST.ckpt：${cpc1_best}" >&2
    exit 1
}

# CPC2 继承 Find_1 模型和辅助监督，但使用 CPC2 损失与冻结策略，并从本轮
# CPC1 BEST 做 model-only 初始化。正式基线继续保留 0.005 warmup 覆盖。
cpc2_overrides=(
    "+experiment=CPC2/Find_1"                    # 选择 Find_1/CPC2 的损失、冻结模块和训练制度。
    "init_from=${cpc1_best}"                      # 仅从本轮 CPC1 BEST 恢复模型权重。
    "${common_overrides[@]}"                      # 保持设备、批量、学习率与密度 cube 参数一致。
    "train.scheduler.warmup_steps=null"           # 允许 warmup_ratio 决定步数，覆盖 CPC2 YAML 的固定 0 步。
    "train.scheduler.warmup_ratio=0.005"          # CPC2 也使用总步数前 0.5% 的 warmup。
    "train.scheduler.warmup_start_factor=0.33"    # 从最大学习率的 33% 线性升高。
    "train.scheduler.patience=1"                  # 连续第 2 次验证无足够改进时降低学习率。
    "train.scheduler.stop_after_lr_reductions=1"  # CPC2 第一次实际降学习率后结束。
)

# 切换运行标识，使 src/train.py 新建 CPC2 目录而不是复用 CPC1 目录。
export POCKET_RUN_STAMP="${cpc2_stamp}"

# 在正式启动前明确打印 CPC1→CPC2 阶段切换和目标目录。
echo "[Find_1] CPC1 完成，启动 CPC2：${cpc2_run}"

# 启动正式 CPC2；一次性 Hydra 覆盖继续保持最高优先级。
python -u src/train.py "${cpc2_overrides[@]}" "$@"

# 训练链完成前再次要求 CPC2 产生 BEST.ckpt。
cpc2_best="${cpc2_run}/checkpoints/BEST.ckpt"

# 缺少 CPC2 BEST 时让任务返回失败，四锁执行器随后进入 try_lock。
[[ -f "${cpc2_best}" ]] || {
    echo "[Find_1][错误] CPC2 没有产生 BEST.ckpt：${cpc2_best}" >&2
    exit 1
}

# 最终成功信息进入 allocation 的 out；allocation 随后创建 try_lock。
echo "[Find_1] CPC1→CPC2 正式训练完成。"
