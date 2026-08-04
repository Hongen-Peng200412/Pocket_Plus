#!/usr/bin/env bash

# 用 CPU 数组并行生成 Stage1 第二版单 PDB BOX pool；本脚本不发布根 _COMPLETE。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

data_root="/storage/penghongen/AdaLigand/Ori_Data"
split_root="${data_root}/stage1_preparation/split"
preparation_root="${data_root}/stage1_preparation_box_pool_2"
output_root="${preparation_root}/box_pool"
state_root="${preparation_root}/run_state/box_pool"
shard_count="${BOX_POOL_SHARD_COUNT:-24}"
shard_index="${SLURM_ARRAY_TASK_ID:?本脚本必须由 Slurm array 启动}"
workers="${SLURM_CPUS_PER_TASK:-8}"

cd "${PROJECT_ROOT}"
python -u ops/box_pool_2/build_box_pool_2.py build-shard \
  --data-root "${data_root}" \
  --train-split "${split_root}/train.json" \
  --validation-split "${split_root}/validation.json" \
  --output-root "${output_root}" \
  --state-root "${state_root}" \
  --shard-count "${shard_count}" \
  --shard-index "${shard_index}" \
  --workers "${workers}"
