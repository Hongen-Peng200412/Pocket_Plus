#!/usr/bin/env bash

# Stage1 V3 calibration 与冻结参数推理入口.
# 本脚本只固定项目环境和配置文件; checkpoint, producer, 清单, 数据划分, 输出根目录及模型代码来源都由提交命令显式传入.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

if (($# == 0)); then
    echo "[stage1_v3][错误] 必须传入 calibrate 或 run 及其全部显式参数." >&2
    exit 2
fi

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli \
    "$@" \
    --config "${PROJECT_ROOT}/configs/inference/stage1_v3.yaml"
