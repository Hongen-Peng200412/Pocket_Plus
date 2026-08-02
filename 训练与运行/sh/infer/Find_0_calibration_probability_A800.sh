#!/usr/bin/env bash
set -euo pipefail

# 本脚本由 submit_task.sh 为每个 Slurm 数组元素启动一次；每个数组元素使用一张 A800。
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
checkpoint="${run_root}/checkpoints/TOP_epoch_00_score_0.2843.ckpt"  # ligand PRAUC 最高的现存 Find_0 checkpoint。
config="${run_root}/config.yaml"                                     # 该训练运行真正落盘且可解析的完整配置。
data_root="/storage/penghongen/AdaLigand/Ori_Data"                    # A–G 正式密度、受体和标签产物根目录。
formal_root="/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${formal_root}/inputs/calibration_pdb_ids.json"             # prepare_inputs 冻结的 100 个唯一 calibration PDB。
output_root="${formal_root}/artifacts"                                # probability、components、centered 与状态标记的共同根目录。

global_shard_count=16                                                  # 整个 calibration 集合固定拆成 16 个互斥分片。
shard_index="${SLURM_ARRAY_TASK_ID:-0}"                               # 数组元素 0 处理第 0 个分片；非数组运行时只处理第 0 个分片。
window_batch_size=8                                                    # 一次完整图滑窗前向包含 8 个 80³ BOX；A800 smoke 已验证。
cache_max_bytes=536870912000                                           # 单进程 Dataset 缓存上限 500 GiB；只按需占用，不预分配。

[[ -f "${formal_root}/inputs/manifest.json" ]] || { echo "缺少冻结输入 manifest" >&2; exit 2; }
[[ -f "${checkpoint}" && -f "${config}" && -f "${pdb_list}" ]] || { echo "正式输入文件不完整" >&2; exit 2; }

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli cal-probability \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" \
    --shard-count "${global_shard_count}" \
    --data-root "${data_root}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --device cuda:0 \
    --window-batch-size "${window_batch_size}" \
    --cache-max-bytes "${cache_max_bytes}" \
    --output-root "${output_root}"
