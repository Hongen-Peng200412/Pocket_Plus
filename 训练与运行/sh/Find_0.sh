#!/usr/bin/env bash

# Find_0 CPC1 从头训练入口；本脚本不再自动接续 CPC2。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
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

export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3}"

devices="${TASK_GPUS:-2}"
nnodes="${TASK_NNODES:-1}"
experiment_group="AdaLigand_Stage1/Find_0/CPC1"
tag="Find_0/CPC1"
run_stamp="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}_CPC1"
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag//\//-}____${run_stamp}"

overrides=(
    "+experiment=CPC1/Find_0"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.ddp_find_unused_parameters=true"
    "train.global_batch_size=48"
    "train.batch_size=8"
    "train.strict_global_batch_size=true"
    "train.enable_batch_size_tuning=false"
    "train.num_workers=16"
    "train.prefetch_factor=4"
    "train.max_epochs=20"
    "train.val_per_epoch=30"
    "train.optimizer.lr=5.0e-5"
    "model.backbone.real_atom_density_cube_size=11"
    "model.backbone.real_density_cube_cfg.cube_size=11"
    "train.scheduler.warmup_ratio=0.005"
    "train.scheduler.patience=2"
    "train.scheduler.stop_after_lr_reductions=4"
    "offline=false"
)

cd "${PROJECT_ROOT}"
echo "[Find_0] 启动 CPC1：${formal_run}"
export TASK_RUN_STAMP="${run_stamp}"
bash "${PROJECT_ROOT}/训练与运行/runtime/launch_training_python.sh" src/train.py "${overrides[@]}" "$@"

formal_best="${formal_run}/checkpoints/BEST.ckpt"
[[ -f "${formal_best}" ]] || {
    echo "[Find_0][错误] CPC1 没有产生 BEST.ckpt：${formal_best}" >&2
    exit 1
}
echo "[Find_0] CPC1 正式训练完成。"
