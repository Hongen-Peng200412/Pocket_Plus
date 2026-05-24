#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_collect_easy20_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_collect_easy20_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J CollectE20
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

RUN_ID="${RUN_ID:?RUN_ID is required}"
MODE="${MODE:?MODE is required: oracle or realistic}"

cd /home/penghongen/My_Project/Pocket_Plus
source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

python Docking/collect_batch_summary.py \
  --run-id "${RUN_ID}" \
  --mode "${MODE}"
