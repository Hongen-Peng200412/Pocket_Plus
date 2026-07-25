#!/usr/bin/env bash

# 任一命令失败立即退出；读取未定义变量时报错；管道中任一命令失败即判失败。
set -euo pipefail

# 本文件只描述 unet_c1 的单阶段正式训练。它不申请 GPU、不创建 release、
# 不实现锁，也不运行 smoke test。通常由 ../submit_task.sh 申请一张 H100 后执行。

# 得到“训练与运行/sh”的绝对位置，使脚本不依赖用户当前工作目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# 当前脚本向上两级是本次 release 的项目根；代码和配置均从该 release 读取。
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

# 项目目录名用于构造默认反馈根；标准 release 内为 Pocket_Plus。
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"

# Conda 根目录可由外部覆盖；默认使用当前用户主目录下的 anaconda3。
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"

# Conda 环境名可由 POCKET_CONDA_ENV 覆盖；默认值是已验证的正式训练环境。
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

# A–G 正式产物根目录。实际调用链为：
# 本变量
# → configs/dataset/stage1_unet_c1.yaml:9
# → Hydra 的 dataset.all_data_path
# → src/train.py UnifiedDataModule.setup()（约第 968–988 行）
# → Stage1Dataset.__init__()（stage1_dataset.py:410–440）
# → _materialize()/__getitem__() 读取密度、受体、配体和辅助标签。
# unet_c1 的模型输入只取 exp_clipnorm_nopost 密度通道，但 Dataset 仍读取受体
# 原子与 ligand_dist.npz，以生成蛋白、核酸和配体距离三项辅助监督。
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"

# Stage1 请求准备产物根目录。实际调用链为：
# 本变量
# → configs/dataset/stage1_unet_c1.yaml:10
# → dataset.box_pool_root=<本目录>/box_pool
# → dataset.split_train=<box_pool>/train
# → dataset.split_val=<box_pool>/validation_selection.npz
# → UnifiedDataModule.setup()
# → Stage1Dataset.__init__() 的 build_request_source()（stage1_dataset.py:470–477）
# → 请求层给出训练与验证使用的 80³ BOX。
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000}"

# 正式训练产物根目录。src/train.py:53–55 读取本变量；
# ExperimentManager._resolve_run_dir()（experiment_manager.py:97–104）建立：
# <本目录>/logs/<experiment_group>/<tag>____<POCKET_RUN_STAMP>/。
# 最终解析配置、源码快照、checkpoint、W&B 和 train.log 都写入这里。
export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"

# allocation_runner.sh 把每节点 GPU 数写入 TASK_GPUS；直接运行时回退为一张卡。
devices="${TASK_GPUS:-1}"

# allocation_runner.sh 把节点数写入 TASK_NNODES；当前正式基线是一台节点。
nnodes="${TASK_NNODES:-1}"

# Lightning 的 train.devices 必须是正整数。
[[ "${devices}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[unet_c1][错误] TASK_GPUS 必须是正整数。" >&2
    exit 2
}

# Lightning 的 train.nnodes 必须是正整数。
[[ "${nnodes}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[unet_c1][错误] TASK_NNODES 必须是正整数。" >&2
    exit 2
}

# conda.sh 可能读取未定义的内部变量，因此激活时暂时关闭 nounset。
set +u
# 在当前 shell 注册 conda activate 函数。
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# 激活包含 PyTorch、Lightning、Hydra 和项目依赖的正式环境。
conda activate "${CONDA_ENV_NAME}"
# 环境激活后恢复未定义变量检查。
set -u

# 优先从本 release 导入 src，避免导入服务器上另一份 Pocket_Plus。
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Hydra 输出完整异常链，便于从 allocation 的 err 定位问题。
export HYDRA_FULL_ERROR=1

# Python 日志不缓存，使训练进度和 traceback 立即写入 out/err。
export PYTHONUNBUFFERED=1

# 每个训练/DataLoader 进程只使用一个 OpenMP 线程，避免 CPU 过度并行。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 限制每个进程中的 MKL 线程数。
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# 限制每个进程中的 OpenBLAS 线程数。
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# 减少 80³ 体素训练中的 CUDA 缓存分配碎片；外部设置优先。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

# Lightning 在一个 Slurm task 内自行建立训练进程。清除父 task 的进程编号，
# 防止它们被误认成已启动的分布式 worker；不删除 GPU 可见性等资源变量。
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID

# Hydra 从项目根下的 configs/ 组合配置，src/train.py 也从该目录启动。
cd "${PROJECT_ROOT}"

# 一张卡时没有 DDP 未使用参数检查；若以后用多卡复用本脚本，则自动开启。
if ((devices > 1)); then
    ddp_find_unused_parameters=true
else
    ddp_find_unused_parameters=false
fi

# 选择完整 unet_c1 experiment，并只覆盖当前正式运行相对 experiment 的参数。
# experiment 继续负责模型、Dataset 与五项损失；五项总损失权重为：
# ligand_area=1.0、ligand_dist=0.3、binding_area=0.1、
# protein_mainchain=0.05、nucleic_mainchain=0.05。
training_overrides=(
    "+experiment=unet_c1"                                   # 选择 density-only U-Net、Stage1 Dataset 与五项损失。
    "init_from=null"                                        # 从头训练，不加载旧 checkpoint。
    "project_name=AdaLigand_Stage1"                         # W&B 项目名。
    "train.devices=${devices}"                              # 每节点 GPU/训练进程数，通常为 1。
    "train.nnodes=${nnodes}"                                # 节点数，当前通常为 1。
    "train.ddp_find_unused_parameters=${ddp_find_unused_parameters}" # 单卡为 false，多卡复用时为 true。
    "train.global_batch_size=48"                            # 每次优化器更新等价处理 48 个 BOX。
    "train.batch_size=6"                                    # 每张 GPU 每次前向处理 6 个 BOX。
    "train.strict_global_batch_size=true"                   # 全局批量必须严格整除实际并行批量。
    "train.enable_batch_size_tuning=false"                  # 禁止自动改变已核定的单卡批量。
    "train.num_workers=20"                                  # 训练 DataLoader 使用 20 个读取进程。
    "train.max_epochs=20"                                   # 最多运行 20 个 epoch。
    "train.val_per_epoch=30"                                # 每个 epoch 等间隔运行 30 次完整验证。
    "train.optimizer.lr=1.0e-4"                             # unet_c1 最大学习率。
    "train.scheduler.warmup_ratio=0.005"                    # 总步数前 0.5% 线性 warmup。
    "train.scheduler.patience=3"                            # 连续第 4 次验证无足够改进时降低学习率。
    "train.scheduler.stop_after_lr_reductions=4"            # 第 4 次实际降学习率后停止。
    "offline=false"                                         # 优先在线记录 W&B。
)

# allocation_runner.sh 通常提供唯一运行标识；直接运行时用当前时间生成。
run_stamp_base="${POCKET_RUN_STAMP:-unet_c1_$(date '+%Y%m%dT%H%M%S')}"

# 单阶段训练仍加 formal 后缀，使目录含义明确且与同一 launch 的其他命令隔离。
formal_stamp="${run_stamp_base}_formal"

# 按 ExperimentManager 的真实目录规则预先定位运行目录，用于碰撞检查与找 BEST。
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/AdaLigand_Stage1-unet_c1/unet_c1____${formal_stamp}"

# 禁止覆盖同名训练目录；碰撞通常表示重复使用了 POCKET_RUN_STAMP。
[[ ! -e "${formal_run}" ]] || {
    echo "[unet_c1][错误] 训练目录已存在：${formal_run}" >&2
    exit 23
}

# 让 src/train.py 使用本次单阶段训练的唯一运行标识。
export POCKET_RUN_STAMP="${formal_stamp}"

# 把即将写入的正式运行目录打印到 allocation 的 out。
echo "[unet_c1] 启动正式训练：${formal_run}"

# 启动正式训练。"$@" 位于固定数组之后，因此提交命令中 `--` 后的 Hydra
# 参数拥有最高优先级，可用于一次实验覆盖而无需修改本文件。
python -u src/train.py "${training_overrides[@]}" "$@"

# 定位本轮由 ModelCheckpoint 保存的正式最佳 checkpoint。
formal_best="${formal_run}/checkpoints/BEST.ckpt"

# 缺少 BEST 表示训练未满足正式完成条件，返回失败让四锁执行器进入 try_lock。
[[ -f "${formal_best}" ]] || {
    echo "[unet_c1][错误] 没有产生 BEST.ckpt：${formal_best}" >&2
    exit 1
}

# 最终成功信息进入 allocation 的 out。
echo "[unet_c1] 正式训练完成。"
