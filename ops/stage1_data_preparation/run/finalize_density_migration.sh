#!/usr/bin/env bash

# 全部迁移分片成功后，复查每个现有来源 NPZ 并最后发布迁移完成标记。
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

data_root="/storage/penghongen/AdaLigand/Ori_Data"
run_root="${data_root}/reports/runs/stage1_npy_migration_20260817_v1"

cd "${project_root}"
python -u ops/stage1_data_preparation/migrate_density_arrays.py finalize \
  --data-root "${data_root}" \
  --run-root "${run_root}" \
  --shard-count 12
