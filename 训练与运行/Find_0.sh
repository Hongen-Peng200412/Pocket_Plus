#!/usr/bin/env bash
set -euo pipefail

scope="/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321743"
runtime_parent="$scope/runtime/My_Project"
shared_repo=/home/penghongen/My_Project/Pocket_Plus
mkdir -p "$runtime_parent/Pocket_Plus"
touch "$runtime_parent/.project-root"
rsync -a   --exclude '.git/'   --exclude '__pycache__/'   --exclude '*.pyc'   --exclude '.pytest_cache/'   "$shared_repo/" "$runtime_parent/Pocket_Plus/"

set +u
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate Pocket_Plus_centos7_cu121_allgpu
set -u
cd "$runtime_parent"

export TRIAL_TAG=replanned_2gpu_chunk2x
export TRIAL_MAX_STEPS=5
export TRIAL_LIMIT_TRAIN_BATCHES=64
bash "$runtime_parent/Pocket_Plus/tmp/stage1_dataset_memory_trial.sh"   "Find_0" "CPC1/Find_0" "2" "8" "10" "$scope"

export FORMAL_RUN_TAG=lr5e5_p2_val30_chunk2x_2gpu_m8
bash "$runtime_parent/Pocket_Plus/tmp/stage1_formal_find_chain.sh"   "$scope" "Find_0" "2" "8" "10"
