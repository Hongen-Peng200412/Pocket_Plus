#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/Pocket_Plus/Ligand/logs/rcsb_ligand_enrichment_%A_%a.out
#SBATCH -e /home/penghongen/My_Project/Pocket_Plus/Ligand/logs/rcsb_ligand_enrichment_%A_%a.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J RCSB_Lig
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4

set -euo pipefail

echo "[Info] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
cd /home/penghongen/My_Project/Pocket_Plus

mkdir -p /home/penghongen/My_Project/Pocket_Plus/Ligand/logs

python -m Ligand.rcsb_enrichment.run_enrichment \
  --raw-json /home/penghongen/My_Project/Data/raw.json \
  --parsed-root /home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb \
  --output-root /storage/penghongen/CIF_Ligand \
  --n-jobs 8 \
  --max-retries 3 \
  --retry-sleep 2.0 \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-count "${SLURM_ARRAY_TASK_COUNT}"

echo "[Info] Array task ${SLURM_ARRAY_TASK_ID} finished."
