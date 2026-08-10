#!/usr/bin/env bash

# 本文件只描述 Find_1 的正式训练。它不申请 GPU、不创建 release、不实现锁，也不运行 smoke test。通常由 ../submit_task.sh 申请1张 H200 后执行。
# CPC1 从头训练；成功产生 BEST.ckpt 后，CPC2 从该 checkpoint 初始化。

# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/train_3/Find_1.sh \
#   --resource h200 \
#   --qos cpu96 \
#   --gpus 1 \
#   --cpus 32 \
#   --mem 1200G \
#   --after_hold \
#   --job-name find1_v2


# ============================================== 一般可以跨任务的通用设置 ==============================================
set -euo pipefail  # 任一命令失败立即退出；读取未定义变量时报错；管道中任一命令失败即判失败。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"  # 得到“训练与运行/sh”的绝对位置，使脚本不依赖用户当前工作目录。
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"  # train_3 向上三级是本次 release 的项目根。
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"  # 项目目录名用于构造默认反馈根；标准 release 内为 Pocket_Plus。
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"  # Conda 根目录可由外部覆盖；默认使用当前用户主目录下的 anaconda3。
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"  # Conda 环境名可由 POCKET_CONDA_ENV 覆盖；默认值是已验证的正式训练环境。

set +u  # conda.sh 可能读取未定义的内部变量，因此激活时暂时关闭 nounset。
source "${CONDA_BASE}/etc/profile.d/conda.sh"  # 在当前 shell 注册 conda activate 函数。
conda activate "${CONDA_ENV_NAME}"  # 激活包含 PyTorch、Lightning、Hydra 和项目依赖的正式环境。
set -u  # 环境激活后恢复未定义变量检查。
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"  # 优先从本 release 导入 src，避免意外导入服务器上其他 Pocket_Plus 副本。
export HYDRA_FULL_ERROR=1  # Hydra 输出完整异常链，便于从 allocation 的 err 定位配置或实例化错误。
export PYTHONUNBUFFERED=1  # Python 日志不缓存，使训练进度和 traceback 立即进入 allocation 的 out/err。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"  # 每个训练/DataLoader 进程只使用一个 OpenMP 线程，避免 CPU 过度并行。
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"  # 限制每个进程中的 MKL 线程数。
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"  # 限制每个进程中的 OpenBLAS 线程数。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"  # 减少 CUDA 缓存分配碎片；外部设置优先。

# Lightning 在当前单一 Slurm task 内自行建立 DDP 进程。清除父 task 的进程编号，防止它们被误认成已经启动的训练 worker；不删除 GPU 可见性等资源变量。
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID











# ======================================================== 运行产物 & wandb ========================================================

# 正式训练产物根目录。src/train.py:53–55 读取本变量；
# ExperimentManager._resolve_run_dir()（experiment_manager.py:97–104）建立：
# <本目录>/logs/<experiment_group>/<tag>____<TASK_RUN_STAMP>/。
export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"
cpc1_experiment_group="AdaLigand_Stage1/Find_1/CPC1"
cpc1_tag="Find_1/CPC1"                                                                # 额外兼任：wandb 项目名下的运行名称
cpc2_experiment_group="AdaLigand_Stage1/Find_1/CPC2"
cpc2_tag="Find_1/CPC2"                                                                # 额外兼任：wandb 项目名下的运行名称

run_stamp_base="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}"
cpc1_stamp="${run_stamp_base}_CPC1"
cpc2_stamp="${run_stamp_base}_CPC2"

# (示例前提：HOME=/home/penghongen、项目目录名为 Pocket_Plus，且外部没有传入 TASK_RUN_STAMP；以 2026-07-26 16:40:33 为启动时间)
# CPC1的运行产物落盘地址: /home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____20260726T164033_CPC1
cpc1_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${cpc1_experiment_group//\//-}/${cpc1_tag//\//-}____${cpc1_stamp}"
# CPC2的运行产物落盘地址: /home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC2/Find_1-CPC2____20260726T164033_CPC2
cpc2_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${cpc2_experiment_group//\//-}/${cpc2_tag//\//-}____${cpc2_stamp}"


project_name="AdaLigand_Stage1"                                                         # wandb 中的项目名











# ================================================================== 任务本身的参数 ==================================================================
# Hydra 以项目根下的 configs/ 组合配置，src/train.py 也从该目录启动。
cd "${PROJECT_ROOT}"

# A–G 正式产物根目录。实际调用链为：
# 本变量
# → configs/dataset/stage1_find.yaml:10
# → Hydra 的 dataset.all_data_path
# → src/train.py UnifiedDataModule.setup()（约第 968–988 行）
# → Stage1Dataset.__init__()（stage1_dataset.py:410–440）
# → _materialize()/__getitem__() 读取各 PDB 的密度、受体、配体和标签，
#   最终物化一个 80×80×80 的训练或验证 BOX。
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"

# Stage1 请求准备产物根目录。实际调用链为：
# 本变量
# → configs/dataset/stage1_find.yaml:11
# → dataset.box_pool_root=<本目录>/box_pool
# → dataset.split_train=<box_pool>/train
# → dataset.split_val=<box_pool>/validation_selection.npz
# → UnifiedDataModule.setup()
# → Stage1Dataset.__init__() 的 build_request_source()（stage1_dataset.py:470–477）
# → 训练请求按 epoch 选择 BOX，验证请求读取固定 selection。
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2}"

#                                  -----------------------------------------------------------------------------------------------                                 #
# --- 卡的申请/参数覆盖 ---
devices="${TASK_GPUS:-1}"  # allocation_runner.sh 把每节点 GPU 数写入 TASK_GPUS；直接运行时回退为两张卡。   # NOTE: 这里需要和 submit_task.sh 命令申请的卡数一致
nnodes="${TASK_NNODES:-1}"  # allocation_runner.sh 把节点数写入 TASK_NNODES；当前正式基线是一台节点。


# 两阶段共同覆盖。命令行数组覆盖 experiment YAML，而调用脚本时最后追加的 "$@" 仍可对某一次实验进行最高优先级覆盖。
common_overrides=(
    "project_name=${project_name}"                        # 覆盖 W&B 项目名。
    "train.devices=${devices}"                            # 每节点 GPU/训练进程数，通常为 2。
    "train.nnodes=${nnodes}"                              # 节点数，当前通常为 1。
    "train.ddp_find_unused_parameters=true"               # Find 条件分支允许某轮存在未用参数。
    "train.global_batch_size=48"                          # 每次优化器更新等价处理 48 个 BOX。
    "train.batch_size=8"                                  # 每张 GPU 每次前向处理 8 个 BOX。
    "train.strict_global_batch_size=false"                # 禁止全局批量对齐逻辑改写脚本显式配置的单卡物理批量。
    "train.enable_batch_size_tuning=false"                # 禁止自动探测改写脚本显式配置的单卡物理批量。
    "train.num_workers=32"                                # 每个训练 DataLoader 使用 32 个读取进程。
    "train.max_epochs=20"                                 # 每个阶段最多运行 20 个 epoch。
    "train.val_per_epoch=5"                               # 按近似固定的训练 BOX 间隔，每个 epoch 等间隔运行 5 次完整验证。
    "train.optimizer.lr=5.0e-5"                           # Find_1 最大学习率。
    "model.backbone.density_cube_cfg.chunk_size=4048"      # 伪原子密度 cube 每块最多处理 4048 个 anchor。
    "model.backbone.real_density_cube_cfg.chunk_size=8192" # 真实受体原子密度 cube 每块最多处理 8192 个原子。
    "offline=false"                                       # 优先把 W&B 在线写入正式项目。
)
# CPC1 选择 Find_1/CPC1 的模型、Dataset、损失和冻结策略，并从头训练。
cpc1_overrides=(
    "+experiment=CPC1/Find_1"                    # 完整选择 Find_1/CPC1 experiment。
    "experiment_group=${cpc1_experiment_group}"  # 覆盖 CPC1 产物的实验分组目录名。
    "tag=${cpc1_tag}"                            # 覆盖 CPC1 运行目录前缀和 W&B 运行名称。
    "init_from=null"                              # 不加载旧 checkpoint。
    "${common_overrides[@]}"                      # 应用两阶段共同覆盖。
    "train.scheduler.warmup_ratio=0.005"          # 总步数前 0.5% 线性 warmup。
    "train.scheduler.patience=3"                  # 连续第 4 次验证无足够改进时降低学习率。
    "train.scheduler.stop_after_lr_reductions=4"  # 第 4 次实际降学习率后结束 CPC1。
)













# ================================================================== 正式启动命令 ==================================================================
echo "[Find_1] 启动 CPC1：${cpc1_run}"
export TASK_RUN_STAMP="${cpc1_stamp}"
#################################### 启动正式 CPC1 ####################################
python -u src/train.py "${cpc1_overrides[@]}" "$@"



# -----------------------------------------------------------------
# CPC2 只使用本轮 CPC1 的 BEST.ckpt。
cpc1_best="${cpc1_run}/checkpoints/BEST.ckpt"
# CPC1 没有正式 BEST 时禁止进入 CPC2。
[[ -f "${cpc1_best}" ]] || {
    echo "[Find_1][错误] CPC1 没有产生 BEST.ckpt：${cpc1_best}" >&2
    exit 1
}
# CPC2 继承 Find_1 主体，但换用 CPC2 损失、冻结策略和短调度制度。
cpc2_overrides=(
    "+experiment=CPC2/Find_1"                    # 选择 Find_1/CPC2 experiment。
    "experiment_group=${cpc2_experiment_group}"  # 覆盖 CPC2 产物的实验分组目录名。
    "tag=${cpc2_tag}"                            # 覆盖 CPC2 运行目录前缀和 W&B 运行名称。
    "init_from=${cpc1_best}"                      # 从本轮 CPC1 BEST 做 model-only 初始化。
    "${common_overrides[@]}"                      # 保持设备、批量、学习率和 cube 参数不变。
    "train.scheduler.warmup_steps=null"           # 允许 warmup_ratio 决定步数，覆盖 CPC2 YAML 的固定 0 步。
    "train.scheduler.warmup_ratio=0.005"          # CPC2 也使用总步数前 0.5% 的 warmup。
    "train.scheduler.warmup_start_factor=0.33"    # 从最大学习率的 33% 线性升高。
    "train.scheduler.patience=1"                  # 连续第 2 次验证无足够改进时降低学习率。
    "train.scheduler.stop_after_lr_reductions=1"  # 第一次实际降学习率后结束 CPC2。
)
export TASK_RUN_STAMP="${cpc2_stamp}"
echo "[Find_1] CPC1 完成，启动 CPC2：${cpc2_run}"
#################################### 启动正式 CPC2 ####################################
python -u src/train.py "${cpc2_overrides[@]}" "$@"
cpc2_best="${cpc2_run}/checkpoints/BEST.ckpt"



# -----------------------------------------------------------------
# 缺少 CPC2 BEST 时返回失败；只有提交时使用 --after_hold 才会进入 try_lock 并保留资源。
[[ -f "${cpc2_best}" ]] || {
    echo "[Find_1][错误] CPC2 没有产生 BEST.ckpt：${cpc2_best}" >&2
    exit 1
}
echo "[Find_1] CPC1→CPC2 正式训练完成。"
