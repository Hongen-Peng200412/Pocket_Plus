#!/usr/bin/env bash
set -euo pipefail

# 固定产生 train 清单的全局分片 1/50：依次落盘完整图概率、组件和 F1_centered；不计算 CLG。
# 常规提交时可在 Pocket_Plus 服务器项目根目录执行：
#
# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_train_F1_shard_01.sh \
#   --resource a800 \
#   --qos cpu96 \
#   --gpus 1 \
#   --cpus 8 \
#   --after_hold \
#   --job-name find0_train_f1_shard01
#
# 本脚本只负责分片 1；它可与其他不重复的 0 至 49 分片并行或分批运行。
# `--resource a800` 选择 nvlink 分区和 A800；`--qos cpu96` 覆盖默认 nvlinkg8。
# `--after_hold` 使任务完成后保留资源，省略它则完成后自动释放。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
PROJECT_ROOT="${POCKET_INFERENCE_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"  # 复用旧 allocation 时可显式绑定已验收的冻结 release。
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
checkpoint="${run_root}/checkpoints/TOP_epoch_00_score_0.2843.ckpt"  # ligand PRAUC 最高的现存 Find_0 checkpoint。
config="${run_root}/config.yaml"                                     # 该 checkpoint 所属运行的完整训练配置。
data_root="/storage/penghongen/AdaLigand/Ori_Data"                    # A–G 正式产物根目录。
inference_root="/storage/penghongen/AdaLigand_stage1_inference"
formal_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${inference_root}/train_pdb_ids.json"                       # 所有模型共用的 13714 个唯一 train PDB。
output_root="${formal_root}/artifacts"

global_shard_count=50                    # train 清单固定拆成 50 个互斥分片。
shard_index=1                            # 本脚本只处理零起始编号中的第 1 号分片。
window_batch_size=10                     # 当前 smoke 验证的保守完整图滑窗批量；正式提交时按可用显存调整。
centered_batch_size=8                    # 正式居中批量；避免 smoke 值 12 仅剩约 0.7 GiB 的显存余量。
cache_max_bytes=107374182400             # 单进程 Dataset 缓存上限 100 GiB。
max_split_events=1                       # 每条候选谱系最多一次拆分。
max_merge_events=1                       # 每条候选谱系最多一次合并。
max_nodes_per_clg=32                     # 单个候选谱系组最多 32 个组件节点。
f1_eligible_limit=200                    # t_F1 层组件数上限；超过时记录拒绝状态而不继续居中前向。

[[ -f "${pdb_list}" && -f "${output_root}/Find_0/calibration/thresholds.json" ]] || { echo "缺少公共 PDB 清单或冻结阈值" >&2; exit 2; }

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli train-produce-prob-f1 \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" \
    --shard-count "${global_shard_count}" \
    --data-root "${data_root}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --device cuda:0 \
    --window-batch-size "${window_batch_size}" \
    --centered-batch-size "${centered_batch_size}" \
    --cache-max-bytes "${cache_max_bytes}" \
    --max-split-events "${max_split_events}" \
    --max-merge-events "${max_merge_events}" \
    --max-nodes-per-clg "${max_nodes_per_clg}" \
    --f1-eligible-limit "${f1_eligible_limit}" \
    --output-root "${output_root}"
