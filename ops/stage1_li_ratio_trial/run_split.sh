#!/usr/bin/env bash
set -euo pipefail

# Li basic_ratio 数据划分入口. 数据划分与三个路径由提交命令显式传入; 本脚本
# 只在隔离目标根目录写入 inputs、artifacts、feedback 与 monitoring.
if (($# != 4)); then
    echo "用法: run_split.sh SPLIT SOURCE_ROOT TARGET_ROOT DATA_ROOT" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
SPLIT="$1"
SOURCE_ROOT="$2"
TARGET_ROOT="$3"
DATA_ROOT="$4"
PRODUCER="unet_c1"
case "${SPLIT}" in
    calibration)
        EXPECTED_COUNT=100
        ;;
    validation)
        EXPECTED_COUNT=200
        ;;
    *)
        echo "[Li trial][错误] SPLIT 必须是 calibration 或 validation." >&2
        exit 2
        ;;
esac

CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

mkdir -p \
    "${TARGET_ROOT}/inputs" \
    "${TARGET_ROOT}/artifacts/${PRODUCER}/${SPLIT}" \
    "${TARGET_ROOT}/artifacts/${PRODUCER}/tuning" \
    "${TARGET_ROOT}/feedback" \
    "${TARGET_ROOT}/monitoring"

exec > >(tee -a "${TARGET_ROOT}/feedback/Li_basic_ratio_${SPLIT}.log") 2>&1

echo "[Li trial] 开始真实复制 ${SPLIT} probability: $(date --iso-8601=seconds)"
rsync -a "${SOURCE_ROOT}/inputs/" "${TARGET_ROOT}/inputs/"
rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='*/probability/***' \
    --include='*/status/probability/***' \
    --exclude='*' \
    "${SOURCE_ROOT}/artifacts/${PRODUCER}/${SPLIT}/" \
    "${TARGET_ROOT}/artifacts/${PRODUCER}/${SPLIT}/"

probability_count="$(
    find "${TARGET_ROOT}/artifacts/${PRODUCER}/${SPLIT}" \
        -path '*/probability/probability_map.npz' -type f | wc -l
)"
complete_count="$(
    find "${TARGET_ROOT}/artifacts/${PRODUCER}/${SPLIT}" \
        -path '*/status/probability/_COMPLETE' -type f | wc -l
)"
if [[ "${probability_count}" != "${EXPECTED_COUNT}" \
    || "${complete_count}" != "${EXPECTED_COUNT}" ]]; then
    echo "[Li trial][错误] ${SPLIT} probability 复制计数不是" \
        "${EXPECTED_COUNT}/${EXPECTED_COUNT}:" \
        "${probability_count}/${complete_count}" >&2
    exit 1
fi
echo "[Li trial] probability 复制完成: probability=${probability_count}, complete=${complete_count}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_ROOT}"
python -u -m ops.stage1_li_ratio_trial.run_trial \
    --phase "${SPLIT}" \
    --config "${TARGET_ROOT}/inputs/stage1_v3.yaml" \
    --pdb-json "${TARGET_ROOT}/inputs/${SPLIT}.json" \
    --data-root "${DATA_ROOT}" \
    --output-root "${TARGET_ROOT}/artifacts" \
    --producer "${PRODUCER}" \
    --split "${SPLIT}" \
    --li-denominator 32768 \
    --prefiltered-min-voxel 8

echo "[Li trial] ${SPLIT} 全流程完成: $(date --iso-8601=seconds)"
