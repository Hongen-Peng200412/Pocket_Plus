#!/usr/bin/env bash

# 由 12×9 CPU 的 Slurm 数组生成 Stage1 v3 单 PDB BOX pool；本脚本不发布根完成标记。
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
preparation_root="${data_root}/stage1_preparation_box_pool_3"
split_root="${preparation_root}/split"
output_root="${preparation_root}/box_pool"
state_root="${preparation_root}/run_state/box_pool"
shard_count=12
shard_index="${SLURM_ARRAY_TASK_ID:?本脚本必须由 Slurm array 启动}"
workers="${SLURM_CPUS_PER_TASK:?Slurm 必须声明每个数组元素的 CPU 数}"

cd "${project_root}"
python -u ops/stage1_data_preparation/build_box_pool_3.py build-shard \
  --data-root "${data_root}" \
  --train-split "${split_root}/train.json" \
  --validation-split "${split_root}/validation.json" \
  --output-root "${output_root}" \
  --state-root "${state_root}" \
  --shard-count "${shard_count}" \
  --shard-index "${shard_index}" \
  --workers "${workers}" \
  --seed 3407
