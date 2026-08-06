#!/usr/bin/env bash
set -euo pipefail

# 从已完成的 Find_0 完整图概率计算逐图 Li 阈值，并在独立根目录生成 Li_centered。
# 本入口不生成 forest、CLG、candidate_eligible 或 Selector 输入；min_voxels 固定为 10。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
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
li_root="/storage/penghongen/AdaLigand_stage1_LI_inference"
probability_output_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts"
output_root="${li_root}/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts"

target_split="calibration"                 # 可选 calibration、validation 或 train。
global_shard_count=1                        # 所有编号合起来覆盖目标清单；可分批提交不同编号。
shard_index="${SLURM_ARRAY_TASK_ID:-0}"     # Slurm 数组编号直接作为全局分片编号。
centered_batch_size=8                       # 每次模型前向的 Li blob BOX 数。
cache_max_bytes=107374182400                # 单进程最多缓存 100 GiB 已读取的 PDB 资产。
min_voxels=10                               # 所有正式推理统一使用的最小连通组件体素数。
max_voxels=2046                             # 沿用 Find_0 calibration 冻结的最大组件体素数。
denominator=32768                           # Li 原始阈值向上量化到相同整数阈值网格。
eligible_limit=200                          # 只用于 `_BLOB_EXCEED` 标示，不改变 calibration 强制产出。

pdb_list="${inference_root}/${target_split}_pdb_ids.json"
[[ -f "${pdb_list}" ]] || { echo "缺少 ${target_split} PDB 清单" >&2; exit 2; }

continue_arguments=()
if [[ "${target_split}" == "calibration" ]]; then
    continue_arguments+=(--continue-on-blob-exceed)
fi

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli produce-li-centered \
    --split "${target_split}" \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" \
    --shard-count "${global_shard_count}" \
    --probability-output-root "${probability_output_root}" \
    --output-root "${output_root}" \
    --data-root "${data_root}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --device cuda:0 \
    --centered-batch-size "${centered_batch_size}" \
    --cache-max-bytes "${cache_max_bytes}" \
    --min-voxels "${min_voxels}" \
    --max-voxels "${max_voxels}" \
    --denominator "${denominator}" \
    --eligible-limit "${eligible_limit}" \
    "${continue_arguments[@]}"
