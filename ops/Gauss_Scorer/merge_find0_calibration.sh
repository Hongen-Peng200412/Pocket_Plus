#!/usr/bin/env bash
set -euo pipefail

# 本脚本只在 8 个参数扫描分片全部成功后运行；它验证82组结果的精确覆盖并冻结唯一 calibration.json。
# 提交前必须显式设置 GAUSS_RESULT_ROOT，使其指向对应父数组 Job 的临时结果目录。
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

result_root="${GAUSS_RESULT_ROOT:?必须显式设置GAUSS_RESULT_ROOT}"
grid_json="${PROJECT_ROOT}/ops/Gauss_Scorer/grid_find0_calibration.json"
calibration_json="/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0/gauss_scorer/calibration.json"

[[ -f "${grid_json}" ]] || { echo "缺少 Gauss 参数网格" >&2; exit 2; }

cd "${PROJECT_ROOT}"
python -u -m ops.Gauss_Scorer.tune merge \
    --grid-json "${grid_json}" \
    --result-root "${result_root}" \
    --calibration-json "${calibration_json}" \
    --producer Find_0
