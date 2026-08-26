#!/usr/bin/env bash

# 复用 F2 语义 blobs, 以 beta=1 调整 basic 参数并评估两个数据划分.
set -euo pipefail

if (($# != 2)); then
    echo "用法: $0 <experiment_root> <data_root>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
STAGE1_ENTRY="${PROJECT_ROOT}/训练与运行/sh/infer/stage1_v3.sh"

EXPERIMENT_ROOT="$1"
DATA_ROOT="$2"
OUTPUT_ROOT="${EXPERIMENT_ROOT}/artifacts"
PRODUCER="unet_c1"
CALIBRATION_JSON="${EXPERIMENT_ROOT}/inputs/calibration.json"
VALIDATION_JSON="${EXPERIMENT_ROOT}/inputs/validation.json"
TUNING_ROOT="${OUTPUT_ROOT}/${PRODUCER}/tuning"
SELECTION_PATH="${TUNING_ROOT}/F2_basic_beta1.json"
FEEDBACK_ROOT="${EXPERIMENT_ROOT}/feedback"

mkdir -p "${FEEDBACK_ROOT}"
exec > >(tee -a "${FEEDBACK_ROOT}/F2_basic_beta1.log") 2>&1

SCRATCH_PARENT="${SLURM_TMPDIR:-/tmp}"
SCRATCH_ROOT="$(mktemp -d "${SCRATCH_PARENT}/stage1_f2_beta1.XXXXXX")"
trap 'rm -rf -- "${SCRATCH_ROOT}"' EXIT

mkdir -p "${SCRATCH_ROOT}/${PRODUCER}"
ln -s "${OUTPUT_ROOT}/${PRODUCER}/calibration" \
    "${SCRATCH_ROOT}/${PRODUCER}/calibration"

echo "[F2 beta1] 开始 calibration basic 调参: $(date --iso-8601=seconds)"
bash "${STAGE1_ENTRY}" tune \
    --producer "${PRODUCER}" \
    --pdb-json "${CALIBRATION_JSON}" \
    --split calibration \
    --output-root "${SCRATCH_ROOT}" \
    --alpha 2 \
    --objective-beta 1 \
    --score-mode basic \
    --prefiltered-min-voxel 8 \
    --data-root "${DATA_ROOT}"

mkdir -p "${TUNING_ROOT}"
TEMP_SELECTION="${TUNING_ROOT}/.F2_basic_beta1.${SLURM_JOB_ID:-$$}.tmp"
cp "${SCRATCH_ROOT}/${PRODUCER}/tuning/F2_basic.json" "${TEMP_SELECTION}"
mv -f "${TEMP_SELECTION}" "${SELECTION_PATH}"
echo "[F2 beta1] 参数发布完成: ${SELECTION_PATH}"
cat "${SELECTION_PATH}"

echo "[F2 beta1] 开始 calibration 评估: $(date --iso-8601=seconds)"
bash "${STAGE1_ENTRY}" evaluate \
    --producer "${PRODUCER}" \
    --pdb-json "${CALIBRATION_JSON}" \
    --split calibration \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --artifact blobs \
    --evaluation-name f2_blobs_basic_beta1_selected \
    --selection-parameters "${SELECTION_PATH}" \
    --data-root "${DATA_ROOT}"

echo "[F2 beta1] 开始 validation 评估: $(date --iso-8601=seconds)"
bash "${STAGE1_ENTRY}" evaluate \
    --producer "${PRODUCER}" \
    --pdb-json "${VALIDATION_JSON}" \
    --split validation \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --artifact blobs \
    --evaluation-name f2_blobs_basic_beta1_selected \
    --selection-parameters "${SELECTION_PATH}" \
    --data-root "${DATA_ROOT}"

cat "${OUTPUT_ROOT}/${PRODUCER}/calibration/evaluation/f2_blobs_basic_beta1_selected.metrics.json"
cat "${OUTPUT_ROOT}/${PRODUCER}/validation/evaluation/f2_blobs_basic_beta1_selected.metrics.json"
echo "[F2 beta1] 全流程完成: $(date --iso-8601=seconds)"
