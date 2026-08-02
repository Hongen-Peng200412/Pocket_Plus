#!/usr/bin/env bash
set -euo pipefail

# 后续补跑入口：复用同一正式目录，跳过已完成的概率、组件和 F1 文件，只增加 CLG_centered。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

run_root="/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-Find_0-CPC1/Find_0-CPC1____job321743_Find_0_CPC1_lr5e5_p2_val30_chunk2x_2gpu_m8_w1"
checkpoint="${run_root}/checkpoints/TOP_epoch_00_score_0.2843.ckpt"
config="${run_root}/config.yaml"
data_root="/storage/penghongen/AdaLigand/Ori_Data"
formal_root="/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${formal_root}/inputs/train_pdb_ids.json"
output_root="${formal_root}/artifacts"

global_shard_count=16                    # 必须与此前 train 分片总数一致。
shard_index="${SLURM_ARRAY_TASK_ID:-0}" # 只补当前全局分片；不同任务不得处理同一编号。
window_batch_size=8                      # 只有缺少旧产物时才会使用；完整分片通常直接跳过概率角色。
centered_batch_size=8                    # A800 正式居中批量。
cache_max_bytes=536870912000             # 单进程 Dataset 缓存上限 500 GiB。
max_split_events=1                       # 每条候选谱系最多一次拆分。
max_merge_events=1                       # 每条候选谱系最多一次合并。
max_nodes_per_clg=32                     # 单个候选谱系组最多 32 个组件节点。
f1_eligible_limit=200                    # t_F1 层候选组件数上限。

[[ -f "${output_root}/Find_0/calibration/thresholds.json" ]] || { echo "缺少冻结阈值" >&2; exit 2; }
cd "${PROJECT_ROOT}"
python -u -m src.inference.cli train-produce-prob-f1-clg \
    --producer Find_0 --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" --shard-count "${global_shard_count}" \
    --data-root "${data_root}" --checkpoint "${checkpoint}" --config "${config}" \
    --device cuda:0 --window-batch-size "${window_batch_size}" \
    --centered-batch-size "${centered_batch_size}" --cache-max-bytes "${cache_max_bytes}" \
    --max-split-events "${max_split_events}" --max-merge-events "${max_merge_events}" \
    --max-nodes-per-clg "${max_nodes_per_clg}" --f1-eligible-limit "${f1_eligible_limit}" \
    --output-root "${output_root}"
