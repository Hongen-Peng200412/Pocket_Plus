#!/usr/bin/env bash

# 使用 Stage1 第二版 BOX pool 从头训练 Find_0 CPC1。
# 本脚本只描述训练本身；GPU、release、launch 与四锁由训练与运行/submit_task.sh 负责。
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

# A–G 完整图与监督标签保持原正式版本不变。
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"

# 第二版准备根与旧 stage1_preparation 并存；Hydra 会读取其 box_pool/ 子目录。
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2}"
export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"

devices="${TASK_GPUS:-2}"
nnodes="${TASK_NNODES:-1}"
run_stamp_base="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}"
run_stamp="${run_stamp_base}_CPC1"
experiment_group="AdaLigand_Stage1/Find_0_box_pool_2/CPC1"
tag="Find_0-box_pool_2/CPC1"
run_dir="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag//\//-}____${run_stamp}"

# H100 每卡批量 6；两卡前向共 12 个 BOX，累积 4 次后保持全局批量 48。
training_overrides=(
    "+experiment=CPC1/Find_0"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.ddp_find_unused_parameters=true"
    "train.global_batch_size=48"
    "train.batch_size=6"
    "train.strict_global_batch_size=true"
    "train.enable_batch_size_tuning=false"
    "train.num_workers=10"
    "train.max_epochs=20"
    "train.val_per_epoch=30"
    "train.optimizer.lr=5.0e-5"
    "train.scheduler.warmup_ratio=0.005"
    "train.scheduler.patience=3"
    "train.scheduler.stop_after_lr_reductions=4"
    "offline=false"
)

echo "[Find_0 box_pool_2] 启动 CPC1：${run_dir}"
export TASK_RUN_STAMP="${run_stamp}"
python -u src/train.py "${training_overrides[@]}" "$@"

best_checkpoint="${run_dir}/checkpoints/BEST.ckpt"
[[ -f "${best_checkpoint}" ]] || {
    echo "[Find_0 box_pool_2][错误] CPC1 没有产生 BEST.ckpt：${best_checkpoint}" >&2
    exit 1
}
echo "[Find_0 box_pool_2] CPC1 正式训练完成。"
