#!/bin/bash
# ============================================================
# Phenix baseline 检查 + 40 扫参 + 110 全量评估
# ============================================================
# 用法:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> sbatch/CPU模板/phenix/run_phenix_eval_after_maps.sbatch
# ============================================================
#SBATCH -o /home/penghongen/My_Project/EVAL_OUT/phenix_/logs/phenix_eval_%j.out
#SBATCH -e /home/penghongen/My_Project/EVAL_OUT/phenix_/logs/phenix_eval_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J phenix-eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
source "${SBATCH_ROOT}/_common.sh"

cd /home/penghongen/My_Project/Pocket_Plus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE_DIR="${PHENIX_BASE_DIR:-/home/penghongen/My_Project/EVAL_OUT/phenix_}"
PHENIX_OUTPUT_ROOT="${PHENIX_OUTPUT_ROOT:-${BASE_DIR}/phenix_diff_maps}"
VAL_JSON="${VAL_JSON:-/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json}"
TEST_JSON="${TEST_JSON:-/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json}"
NUM_SHARDS="${PHENIX_NUM_SHARDS:-30}"

mkdir -p "${BASE_DIR}/logs"

echo "[phenix-eval] job_id=${SLURM_JOB_ID}"
echo "[phenix-eval] base_dir=${BASE_DIR}"
echo "[phenix-eval] phenix_output_root=${PHENIX_OUTPUT_ROOT}"

srun python src/inference/baseline/check_phenix_diff_maps.py \
    --phenix_output_root "${PHENIX_OUTPUT_ROOT}" \
    --num_shards "${NUM_SHARDS}" \
    --val_json "${VAL_JSON}" \
    --test_json "${TEST_JSON}" \
    --systems stardard strict

srun python src/inference/main/run_baseline_two_stage.py \
    --base_dir "${BASE_DIR}" \
    --phenix_output_root "${PHENIX_OUTPUT_ROOT}" \
    --val_json "${VAL_JSON}" \
    --test_json "${TEST_JSON}" \
    --systems stardard strict \
    --baselines phenix_real_space_diff_map \
    --device cpu

echo "[phenix-eval] done."
