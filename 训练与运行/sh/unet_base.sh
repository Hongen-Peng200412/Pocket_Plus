#!/usr/bin/env bash

# unet_base 正式训练入口; 使用 56 个密度通道从头训练 density-only RAUNet64。
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

devices="${TASK_GPUS:-1}"
nnodes="${TASK_NNODES:-1}"
resource_type="${TASK_RESOURCE_TYPE:-a800}"
cpu_count="${TASK_CPU_COUNT:-16}"
if [[ "${resource_type}" != "a800" || "${nnodes}" != "1" || "${devices}" != "1" || "${cpu_count}" != "16" ]]; then
    echo "[unet_base][错误] 本任务要求 resource=a800, nodes=1, gpus=1, cpus=16; 实际为 resource=${resource_type}, nodes=${nnodes}, gpus=${devices}, cpus=${cpu_count}。" >&2
    exit 2
fi
ddp_find_unused_parameters=false

experiment_group="AdaLigand_Stage1/unet_base"
tag="unet_base"
run_stamp="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}_formal"
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag}____${run_stamp}"

overrides=(
    "+experiment=unet_base"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.ddp_find_unused_parameters=${ddp_find_unused_parameters}"
    "train.global_batch_size=48"
    "train.batch_size=8"
    "train.strict_global_batch_size=true"
    "train.enable_batch_size_tuning=false"
    "train.num_workers=16"
    "train.prefetch_factor=4"
    "train.max_epochs=20"
    "train.val_per_epoch=40"
    "train.optimizer.lr=1.0e-4"
    "train.scheduler.warmup_ratio=0.005"
    "train.scheduler.patience=3"
    "train.scheduler.stop_after_lr_reductions=3"
    "offline=false"
)

cd "${PROJECT_ROOT}"
echo "[unet_base] 启动正式训练: ${formal_run}"
export TASK_RUN_STAMP="${run_stamp}"
python -u src/train.py "${overrides[@]}" "$@"

formal_best="${formal_run}/checkpoints/BEST.ckpt"
[[ -f "${formal_best}" ]] || {
    echo "[unet_base][错误] 没有产生 BEST.ckpt: ${formal_best}" >&2
    exit 1
}
echo "[unet_base] 正式训练完成。"
