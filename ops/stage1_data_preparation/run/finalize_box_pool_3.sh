#!/usr/bin/env bash

# 全部 BOX pool 分片成功后，发布 manifest、验证选择、配置、摘要和完成标记。
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
preparation_root="${data_root}/stage1_preparation_box_pool_3"
split_root="${preparation_root}/split"

cd "${project_root}"
python -u ops/stage1_data_preparation/build_box_pool_3.py finalize \
  --data-root "${data_root}" \
  --train-split "${split_root}/train.json" \
  --validation-split "${split_root}/validation.json" \
  --output-root "${preparation_root}/box_pool" \
  --state-root "${preparation_root}/run_state/box_pool" \
  --shard-count 12 \
  --seed 3407
