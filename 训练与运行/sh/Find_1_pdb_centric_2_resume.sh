#!/usr/bin/env bash

# Find_1 PDB-centric-2 从完整 validation checkpoint 恢复训练。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

source_checkpoint="${FIND1_RESUME_SOURCE_CHECKPOINT:?必须设置 FIND1_RESUME_SOURCE_CHECKPOINT}"
source_sha256="${FIND1_RESUME_SOURCE_SHA256:?必须设置 FIND1_RESUME_SOURCE_SHA256}"
resume_epoch="${FIND1_RESUME_EPOCH:?必须设置 FIND1_RESUME_EPOCH}"
resume_skip_train_batches="${FIND1_RESUME_SKIP_TRAIN_BATCHES:?必须设置 FIND1_RESUME_SKIP_TRAIN_BATCHES}"

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
experiment_group="AdaLigand_Stage1_pdb_centric_resume/Find_1/pdb_centric_2"
tag="Find_1/pdb_centric_2_resume"
run_stamp="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}_pdb_centric_2_resume"
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag//\//-}____${run_stamp}"
resume_checkpoint="${formal_run}/checkpoints/resume_state.ckpt"

cd "${PROJECT_ROOT}"
python ops/find1_historical_resume/rebase_checkpoint.py \
    --source "${source_checkpoint}" \
    --destination "${formal_run}/checkpoints" \
    --expected-source-sha256 "${source_sha256}"

overrides=(
    "+experiment=CPC1/Find_1_pdb_centric_2"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "resume_from_checkpoint=${resume_checkpoint}"
    "+train.resume_skip_train_batches=${resume_skip_train_batches}"
    "+train.resume_skip_epoch=${resume_epoch}"
    "+train.resume_after_completed_validation=true"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.global_batch_size=48"
    "train.batch_size=6"
    "train.num_workers=30"
    "train.prefetch_factor=4"
    "train.optimizer.lr=5.0e-5"
    "train.scheduler.patience=3"
    "train.scheduler.stop_after_lr_reductions=3"
    "model.backbone.density_cube_cfg.chunk_size=4096"
    "model.backbone.real_density_cube_cfg.chunk_size=8192"
    "model.backbone.real_atom_density_cube_size=9"
    "model.backbone.real_density_cube_cfg.cube_size=9"
    "offline=false"
)

echo "[Find_1_pdb_centric_2_resume] 从 ${source_checkpoint} 恢复至 ${formal_run}"
export TASK_RUN_STAMP="${run_stamp}"
bash "${PROJECT_ROOT}/训练与运行/runtime/launch_training_python.sh" src/train.py "${overrides[@]}" "$@"

formal_best="${formal_run}/checkpoints/BEST.ckpt"
[[ -f "${formal_best}" ]] || {
    echo "[Find_1_pdb_centric_2_resume][错误] 没有产生 BEST.ckpt：${formal_best}" >&2
    exit 1
}
echo "[Find_1_pdb_centric_2_resume] PDB-centric-2 恢复训练完成。"
