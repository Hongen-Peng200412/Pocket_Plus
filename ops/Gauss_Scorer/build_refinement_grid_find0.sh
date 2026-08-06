#!/usr/bin/env bash
set -euo pipefail

# 从第一阶段 calibration.json 的 selected_parameters 生成 5×5×1×15 精修网格。
# 输出属于一次参数搜索的临时输入，不写入正式 Stage1 产物目录。
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

phase1_calibration_json="${GAUSS_PHASE1_CALIBRATION_JSON:?必须设置GAUSS_PHASE1_CALIBRATION_JSON}"
output_grid_json="${GAUSS_REFINEMENT_GRID_JSON:?必须设置GAUSS_REFINEMENT_GRID_JSON}"

cd "${PROJECT_ROOT}"
python -u -m ops.Gauss_Scorer.tune build-refinement-grid \
    --phase1-calibration-json "${phase1_calibration_json}" \
    --output-grid-json "${output_grid_json}"
