#!/usr/bin/env bash
set -euo pipefail

# 本脚本只扫描 Find_0 calibration 的 Gauss scorer 参数，不修改 forest.npz 或 GPU 主线产物。
# 推荐提交为 8 项 CPU 数组；每项读取同一批 86 个可评估 PDB，并按配置编号取模承接约 10 组参数。
# 中间结果只写入 /storage/penghongen/tmp/find0_gauss_scorer_job<父数组JobID>/parts/。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

task_index="${SLURM_ARRAY_TASK_ID:?本脚本必须由Slurm数组任务运行}"
task_count="${GAUSS_TASK_COUNT:-8}"
array_job_id="${SLURM_ARRAY_JOB_ID:?缺少父数组Job ID}"

data_root="/storage/penghongen/AdaLigand/Ori_Data"
inference_root="/storage/penghongen/AdaLigand_stage1_inference"
formal_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${inference_root}/calibration_pdb_ids.json"
output_root="${formal_root}/artifacts"
grid_json="${PROJECT_ROOT}/ops/Gauss_Scorer/grid_find0_calibration.json"
result_root="${GAUSS_RESULT_ROOT:-/storage/penghongen/tmp/find0_gauss_scorer_job${array_job_id}}"

[[ -f "${pdb_list}" && -f "${grid_json}" ]] || { echo "缺少 calibration 清单或 Gauss 参数网格" >&2; exit 2; }
mkdir -p "${result_root}/parts"

cd "${PROJECT_ROOT}"
python -u -m ops.Gauss_Scorer.tune evaluate-shard \
    --pdb-list "${pdb_list}" \
    --data-root "${data_root}" \
    --output-root "${output_root}" \
    --producer Find_0 \
    --grid-json "${grid_json}" \
    --result-root "${result_root}" \
    --task-index "${task_index}" \
    --task-count "${task_count}"
