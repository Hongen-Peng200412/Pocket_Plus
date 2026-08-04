#!/usr/bin/env bash

# 使用 Stage1 第二版 BOX pool 从头训练 unet_c1；本任务只有一个正式训练阶段。
# GPU、release、launch 与四锁由训练与运行/submit_task.sh 负责。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID

cd "${PROJECT_ROOT}"

# A–G 完整图与监督标签不变；只有 BOX 请求来源切换到第二版准备根。
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2}"
export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"

devices="${TASK_GPUS:-1}"
nnodes="${TASK_NNODES:-1}"
run_stamp_base="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}"
run_stamp="${run_stamp_base}_formal"
experiment_group="AdaLigand_Stage1/unet_c1_box_pool_2"
tag="unet_c1_box_pool_2"
run_dir="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag//\//-}____${run_stamp}"

if ((devices > 1)); then
    ddp_find_unused_parameters=true
else
    ddp_find_unused_parameters=false
fi

# 保留现有 unet_c1 的模型、损失、批量、学习率、验证频率与停止制度。
training_overrides=(
    "+experiment=unet_c1"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.ddp_find_unused_parameters=${ddp_find_unused_parameters}"
    "train.global_batch_size=48"
    "train.batch_size=6"
    "train.strict_global_batch_size=true"
    "train.enable_batch_size_tuning=false"
    "train.num_workers=20"
    "train.max_epochs=20"
    "train.val_per_epoch=30"
    "train.optimizer.lr=1.0e-4"
    "train.scheduler.warmup_ratio=0.005"
    "train.scheduler.patience=3"
    "train.scheduler.stop_after_lr_reductions=4"
    "offline=false"
)

echo "[unet_c1 box_pool_2] 启动正式训练：${run_dir}"
export TASK_RUN_STAMP="${run_stamp}"
python -u src/train.py "${training_overrides[@]}" "$@"

best_checkpoint="${run_dir}/checkpoints/BEST.ckpt"
[[ -f "${best_checkpoint}" ]] || {
    echo "[unet_c1 box_pool_2][错误] 没有产生 BEST.ckpt：${best_checkpoint}" >&2
    exit 1
}
echo "[unet_c1 box_pool_2] 正式训练完成。"
