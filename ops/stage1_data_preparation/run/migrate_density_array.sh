#!/usr/bin/env bash

# 由 12×9 CPU 的 Slurm 数组迁移四类完整体数组；本脚本不发布根完成标记。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../../.." && pwd -P)"
conda_base="${CONDA_BASE:-${HOME}/anaconda3}"
conda_env_name="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${conda_env_name}"
set -u

export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

data_root="/storage/penghongen/AdaLigand/Ori_Data"
run_root="${data_root}/reports/runs/stage1_npy_migration_20260817_v1"
shard_count=12
shard_index="${SLURM_ARRAY_TASK_ID:?本脚本必须由 Slurm array 启动}"
workers="${SLURM_CPUS_PER_TASK:?Slurm 必须声明每个数组元素的 CPU 数}"

cd "${project_root}"
python -u ops/stage1_data_preparation/migrate_density_arrays.py migrate-shard \
  --data-root "${data_root}" \
  --run-root "${run_root}" \
  --shard-count "${shard_count}" \
  --shard-index "${shard_index}" \
  --workers "${workers}"
