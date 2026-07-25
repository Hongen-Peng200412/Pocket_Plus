#!/bin/bash
# ============================================================
# 一键提交 Phenix 生成 array + dependency 评估
# ============================================================
# 用法:
#   bash sbatch/CPU模板/phenix/submit_phenix_pipeline.sh
# ============================================================

set -euo pipefail

cd /home/penghongen/My_Project/Pocket_Plus

BASE_DIR="${PHENIX_BASE_DIR:-/home/penghongen/My_Project/EVAL_OUT/phenix_}"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

GEN_SCRIPT="sbatch/CPU模板/phenix/generate_phenix_maps_array.sbatch"
EVAL_SCRIPT="sbatch/CPU模板/phenix/run_phenix_eval_after_maps.sbatch"

GEN_JOB_RAW="$(sbatch --parsable "${GEN_SCRIPT}")"
GEN_JOB_ID="${GEN_JOB_RAW%%;*}"
EVAL_JOB_RAW="$(sbatch --parsable --dependency=afterok:${GEN_JOB_ID} "${EVAL_SCRIPT}")"
EVAL_JOB_ID="${EVAL_JOB_RAW%%;*}"

echo "[submit_phenix_pipeline] generate_array_job=${GEN_JOB_RAW}"
echo "[submit_phenix_pipeline] eval_job=${EVAL_JOB_RAW}"
echo "[submit_phenix_pipeline] logs=${LOG_DIR}"
