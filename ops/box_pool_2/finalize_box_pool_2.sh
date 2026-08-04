#!/usr/bin/env bash

# 全部分片成功后，验收集合并原子发布 Stage1 第二版 BOX pool 根契约。
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

data_root="/storage/penghongen/AdaLigand/Ori_Data"
split_root="${data_root}/stage1_preparation/split"
preparation_root="${data_root}/stage1_preparation_box_pool_2"
shard_count="${BOX_POOL_SHARD_COUNT:-24}"

cd "${PROJECT_ROOT}"
python -u ops/box_pool_2/build_box_pool_2.py finalize \
  --data-root "${data_root}" \
  --train-split "${split_root}/train.json" \
  --validation-split "${split_root}/validation.json" \
  --output-root "${preparation_root}/box_pool" \
  --state-root "${preparation_root}/run_state/box_pool" \
  --shard-count "${shard_count}"
