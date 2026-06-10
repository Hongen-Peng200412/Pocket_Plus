#!/bin/bash
# ============================================================
# Non-DL density-channel baseline: posdiff_clipnorm_DoG2
# Runs both systems: stardard + strict
# ============================================================
# Usage:
#   sbatch sbatch/CPU模板/non_DL/run_posdiff_clipnorm_DoG2.sbatch
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/non_dl_posdiff_clipnorm_DoG2_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/non_dl_posdiff_clipnorm_DoG2_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J ndl-dog2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
source "${SBATCH_ROOT}/_common.sh"

cd /home/penghongen/My_Project/Pocket_Plus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE_DIR="${NON_DL_BASE_DIR:-/home/penghongen/My_Project/EVAL_OUT/non_DL}"
VAL_JSON="${VAL_JSON:-/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json}"
TEST_JSON="${TEST_JSON:-/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json}"

mkdir -p "${BASE_DIR}/logs"

echo "[non-dl] job_id=${SLURM_JOB_ID}"
echo "[non-dl] baseline=posdiff_clipnorm_DoG2"
echo "[non-dl] systems=stardard strict"
echo "[non-dl] base_dir=${BASE_DIR}"
echo "[non-dl] cache_n_jobs=8 cache_backend=loky"

srun python src/inference/main/run_baseline_two_stage.py \
    --base_dir "${BASE_DIR}" \
    --val_json "${VAL_JSON}" \
    --test_json "${TEST_JSON}" \
    --systems stardard strict \
    --baselines posdiff_clipnorm_DoG2 \
    --device cpu \
    --cache_n_jobs 8 \
    --cache_backend loky

echo "[non-dl] posdiff_clipnorm_DoG2 done."
