#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_easy20_prescan_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_easy20_prescan_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J Easy20Scan
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

PRESCAN_RUN_ID="${PRESCAN_RUN_ID:-20260521_easy20_prescan_v1}"

cd /home/penghongen/My_Project/Pocket_Plus
source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

python Docking/run_easy20_prescan.py \
  --prescan-run-id "${PRESCAN_RUN_ID}" \
  --top-n 20 \
  --merge-center-distance 8.0 \
  --large-heavy-atoms 80
