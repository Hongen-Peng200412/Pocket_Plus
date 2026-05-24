#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_realistic_parallel_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_realistic_parallel_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J RealParallel
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

RUN_ID="${RUN_ID:?RUN_ID is required}"
SAMPLE_ID="${SAMPLE_ID:?SAMPLE_ID is required}"
NSTRUCT="${NSTRUCT:-5}"
ROSETTA_JOBS="${ROSETTA_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

cd /home/penghongen/My_Project/Pocket_Plus
source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SHARD_ID="parallel_rewrite_${SAMPLE_ID}_${SLURM_JOB_ID:-manual}"

python Docking/run_docking_batch.py \
  --run-id "${RUN_ID}" \
  --sample-list "${SAMPLE_ID}" \
  --nstruct "${NSTRUCT}" \
  --jobs 1 \
  --rosetta-jobs "${ROSETTA_JOBS}" \
  --shard-id "${SHARD_ID}" \
  --min-voxels 1 \
  --merge-min-voxel-distance 1000000000 \
  --merge-center-distance 8.0 \
  --merge-max-bbox-span-increase 1000000000
