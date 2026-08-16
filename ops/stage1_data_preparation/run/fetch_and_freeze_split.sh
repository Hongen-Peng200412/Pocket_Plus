#!/usr/bin/env bash

# 建立 EMDB 发布时间缓存，再冻结日期、质量、资产与 200/100/剩余划分。
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
candidate_path="${data_root}/reports/runs/adaligand_ag_20260711T154658/stage_g_analysis/candidates.pending.jsonl"
pair_list_path="${data_root}/raw/pair_list.jsonl"
preparation_root="${data_root}/stage1_preparation_box_pool_3"
split_root="${preparation_root}/split"
release_cache="${split_root}/emdb_release_dates.jsonl"

cd "${project_root}"
python -u ops/stage1_data_preparation/freeze_split.py fetch-release-dates \
  --candidates "${candidate_path}" \
  --pair-list "${pair_list_path}" \
  --output "${release_cache}" \
  --workers 32 \
  --timeout-seconds 30 \
  --retry-count 4 \
  --fsync-interval 100

python -u ops/stage1_data_preparation/freeze_split.py freeze \
  --candidates "${candidate_path}" \
  --pair-list "${pair_list_path}" \
  --release-cache "${release_cache}" \
  --data-root "${data_root}" \
  --output-root "${split_root}" \
  --seed 3407
