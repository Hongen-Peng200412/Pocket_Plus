#!/usr/bin/env bash

# Find_1 PDB-centric-2 从头训练入口; 本轮只完成配置与测试, 未经新授权不提交.
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
experiment_group="AdaLigand_Stage1_pdb_centric/Find_1/pdb_centric_2"
tag="Find_1/pdb_centric_2"
run_stamp="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}_pdb_centric_2"
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag//\//-}____${run_stamp}"

overrides=(
    "+experiment=CPC1/Find_1_pdb_centric_2"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "model.backbone.density_cube_cfg.chunk_size=4096"
    "model.backbone.real_density_cube_cfg.chunk_size=8192"
    "model.backbone.real_atom_density_cube_size=9"
    "model.backbone.real_density_cube_cfg.cube_size=9"
    "offline=false"
)

cd "${PROJECT_ROOT}"
echo "[Find_1_pdb_centric_2] 启动 PDB-centric-2：${formal_run}"
export TASK_RUN_STAMP="${run_stamp}"
bash "${PROJECT_ROOT}/训练与运行/runtime/launch_training_python.sh" src/train.py "${overrides[@]}" "$@"

formal_best="${formal_run}/checkpoints/BEST.ckpt"
[[ -f "${formal_best}" ]] || {
    echo "[Find_1_pdb_centric_2][错误] 没有产生 BEST.ckpt：${formal_best}" >&2
    exit 1
}
echo "[Find_1_pdb_centric_2] PDB-centric-2 正式训练完成。"
