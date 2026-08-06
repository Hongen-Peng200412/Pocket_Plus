#!/usr/bin/env bash
set -euo pipefail

# 为已有 Find_0 components 添油式生成一个 F_alpha-centered 文件，不重建 forest 或 CLG。
# 修改 target_split、alpha、global_shard_count，并提交需要的数组编号即可。
# alpha 只允许 1/2、2/3、4/5、1/1、5/4、3/2、2/1；1/1 对应历史 F1_centered。
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
output_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts"

target_split="calibration"                 # 可选 calibration、validation 或 train。
alpha="2/3"                                # 每次只生成一个冻结 F_alpha 阈值对应的 centered 文件。
global_shard_count=1                        # 所有编号合起来覆盖目标清单；可只提交部分编号后续跑。
shard_index="${SLURM_ARRAY_TASK_ID:-0}"     # Slurm 数组编号直接作为全局分片编号。
centered_batch_size=8                       # 每次模型前向的 80³ BOX 数，按实际显存调整。
cache_max_bytes=107374182400                # 单进程最多缓存 100 GiB 已读取的 PDB 资产。

pdb_list="${inference_root}/${target_split}_pdb_ids.json"
[[ -f "${pdb_list}" ]] || { echo "缺少 ${target_split} PDB 清单" >&2; exit 2; }

continue_arguments=()
if [[ "${target_split}" == "calibration" ]]; then
    continue_arguments+=(--continue-on-blob-exceed)
fi

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli produce-falpha \
    --split "${target_split}" \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --shard-index "${shard_index}" \
    --shard-count "${global_shard_count}" \
    --alpha "${alpha}" \
    --data-root "${data_root}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --device cuda:0 \
    --centered-batch-size "${centered_batch_size}" \
    --cache-max-bytes "${cache_max_bytes}" \
    --output-root "${output_root}" \
    "${continue_arguments[@]}"
