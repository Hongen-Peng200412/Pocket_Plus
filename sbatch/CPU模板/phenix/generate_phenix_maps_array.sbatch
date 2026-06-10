#!/bin/bash
# ============================================================
# Phenix 差图预生成: 30 个 array task, 每个 task 1 核 CPU
# ============================================================
# 用法:
#   sbatch sbatch/CPU模板/phenix/generate_phenix_maps_array.sbatch
# ============================================================
#SBATCH -o /home/penghongen/My_Project/EVAL_OUT/phenix_/logs/phenix_generate_%A_%a.out
#SBATCH -e /home/penghongen/My_Project/EVAL_OUT/phenix_/logs/phenix_generate_%A_%a.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J phenix-map
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-29

set -euo pipefail

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
source "${SBATCH_ROOT}/_common.sh"

cd /home/penghongen/My_Project/Pocket_Plus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PHENIX_OUTPUT_ROOT="${PHENIX_OUTPUT_ROOT:-/home/penghongen/My_Project/EVAL_OUT/phenix_/phenix_diff_maps}"
PHENIX_BIN="${PHENIX_BIN:-/home/yangjy/software/phenix/build/bin/phenix.real_space_diff_map}"
RESOLUTION_CSV="${RESOLUTION_CSV:-/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv}"
VAL_JSON="${VAL_JSON:-/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json}"
TEST_JSON="${TEST_JSON:-/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json}"

mkdir -p "${PHENIX_OUTPUT_ROOT}" "$(dirname "${PHENIX_OUTPUT_ROOT}")/logs"

echo "[phenix-map] job_id=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "[phenix-map] output_root=${PHENIX_OUTPUT_ROOT}"

srun python src/inference/baseline/generate_phenix_diff_maps.py \
    --phenix_output_root "${PHENIX_OUTPUT_ROOT}" \
    --phenix_bin "${PHENIX_BIN}" \
    --resolution_csv "${RESOLUTION_CSV}" \
    --val_json "${VAL_JSON}" \
    --test_json "${TEST_JSON}" \
    --systems stardard strict

echo "[phenix-map] array task done."
