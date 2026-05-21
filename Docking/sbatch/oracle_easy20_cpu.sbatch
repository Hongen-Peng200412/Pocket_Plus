#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_oracle_easy20_%A_%a.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_oracle_easy20_%A_%a.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J OracleE20
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-19%12

RUN_ID="${RUN_ID:-20260521_oracle_easy20_nk_nstruct5_v1}"
PRESCAN_RUN_ID="${PRESCAN_RUN_ID:-20260521_easy20_prescan_v1}"
SAMPLE_LIST="${SAMPLE_LIST:-/home/penghongen/分子对接尝试/easy20_prescans/${PRESCAN_RUN_ID}/easy20_by_nk.txt}"
NSTRUCT="${NSTRUCT:-5}"
JOBS="${JOBS:-1}"
TASKS="${TASKS:-true_center_identity,offset_center_identity,true_center_hungarian,offset_center_hungarian}"
RECEPTORS="${RECEPTORS:-true_receptor,cryoatom_receptor}"
OFFSET_RADII="${OFFSET_RADII:-4,6,8}"
OFFSET_SEEDS="${OFFSET_SEEDS:-3}"

cd /home/penghongen/My_Project/Pocket_Plus
source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mapfile -t SAMPLES < <(grep -v '^[[:space:]]*$' "${SAMPLE_LIST}")
SAMPLE_ID="${SAMPLES[${SLURM_ARRAY_TASK_ID:-0}]}"
if [[ -z "${SAMPLE_ID}" ]]; then
  echo "No sample for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}; exiting."
  exit 0
fi
SHARD_ID="${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_${SAMPLE_ID}"

python Docking/run_oracle_docking_batch.py \
  --run-id "${RUN_ID}" \
  --sample-list "${SAMPLE_ID}" \
  --tasks "${TASKS}" \
  --offset-radii "${OFFSET_RADII}" \
  --offset-seeds "${OFFSET_SEEDS}" \
  --nstruct "${NSTRUCT}" \
  --jobs "${JOBS}" \
  --receptors "${RECEPTORS}" \
  --shard-id "${SHARD_ID}"
