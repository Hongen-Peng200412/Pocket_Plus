#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_oracle_parallel_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_oracle_parallel_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J OraParallel
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

RUN_ID="${RUN_ID:?RUN_ID is required}"
SAMPLE_ID="${SAMPLE_ID:?SAMPLE_ID is required}"
NSTRUCT="${NSTRUCT:-5}"
ROSETTA_JOBS="${ROSETTA_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
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

SHARD_ID="parallel_rewrite_${SAMPLE_ID}_${SLURM_JOB_ID:-manual}"

python Docking/run_oracle_docking_batch.py \
  --run-id "${RUN_ID}" \
  --sample-list "${SAMPLE_ID}" \
  --tasks "${TASKS}" \
  --offset-radii "${OFFSET_RADII}" \
  --offset-seeds "${OFFSET_SEEDS}" \
  --nstruct "${NSTRUCT}" \
  --jobs 1 \
  --rosetta-jobs "${ROSETTA_JOBS}" \
  --receptors "${RECEPTORS}" \
  --shard-id "${SHARD_ID}"
