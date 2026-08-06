#!/usr/bin/env bash
set -euo pipefail

# 后续补跑入口：复用同一正式目录，跳过已完成的概率、组件和 F1 文件，只增加 CLG_centered。
# 在 Pocket_Plus 服务器项目根目录提交两个 calibration 分片：
#
# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_calibration_CLG.sh \
#   --resource a800 \
#   --qos cpu96 \
#   --gpus 1 \
#   --cpus 8 \
#   --array '0-1' \
#   --after_hold \
#   --job-name find0_cal_clg
#
# 本脚本和 probability、F1 脚本使用相同的两份 PDB 归属，可在以后独立续跑 CLG-centered。
# `--resource a800` 选择 nvlink 分区和 A800；`--qos cpu96` 覆盖默认 nvlinkg8。
# `--after_hold` 使每个数组元素完成后保留资源，省略它则完成后自动释放。
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
inference_root="/storage/penghongen/AdaLigand_stage1_inference"
formal_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${inference_root}/calibration_pdb_ids.json"
output_root="${formal_root}/artifacts"

global_shard_count=2                     # 必须与 calibration probability 和 F1 的两分片总数一致。
shard_index="${SLURM_ARRAY_TASK_ID:-0}" # 只补当前全局分片；不同任务不得处理同一编号。
centered_batch_size=8                    # 当前 smoke 验证的保守居中批量；正式提交时按可用显存调整。
cache_max_bytes=107374182400             # 单进程 Dataset 缓存上限 100 GiB。
max_split_events=1                       # 每条候选谱系最多一次拆分。
max_merge_events=1                       # 每条候选谱系最多一次合并。
max_nodes_per_clg=32                     # 单个候选谱系组最多 32 个组件节点。
f1_eligible_limit=200                    # t_F1 层候选组件数上限。

[[ -f "${output_root}/Find_0/calibration/thresholds.json" ]] || { echo "缺少冻结阈值" >&2; exit 2; }
cd "${PROJECT_ROOT}"
python -u -m src.inference.cli cal-produce-f1-clg \
    --producer Find_0 --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" --shard-count "${global_shard_count}" \
    --data-root "${data_root}" --checkpoint "${checkpoint}" --config "${config}" \
    --device cuda:0 --centered-batch-size "${centered_batch_size}" \
    --cache-max-bytes "${cache_max_bytes}" \
    --max-split-events "${max_split_events}" --max-merge-events "${max_merge_events}" \
    --max-nodes-per-clg "${max_nodes_per_clg}" --f1-eligible-limit "${f1_eligible_limit}" \
    --output-root "${output_root}"
