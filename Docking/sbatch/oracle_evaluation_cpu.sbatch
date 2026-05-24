#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_oracle_eval_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_oracle_eval_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J OracleEval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

EVAL_RUN_ID="${EVAL_RUN_ID:?EVAL_RUN_ID is required}"
ORACLE_RUN_ID="${ORACLE_RUN_ID:?ORACLE_RUN_ID is required}"

cd /home/penghongen/My_Project/Pocket_Plus
source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python Docking/evaluation/run_oracle_evaluation.py \
  --eval-run-id "${EVAL_RUN_ID}" \
  --oracle-run-id "${ORACLE_RUN_ID}"
