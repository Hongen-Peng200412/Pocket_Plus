#!/bin/bash
# ============================================================
# A800 单卡训练 (1 GPU, 24 核 CPU)
# ============================================================
# 用法: sbatch sbatch/a800/1gpu.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p a800
#SBATCH --qos=a800g2
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
source "${SCRIPT_DIR}/../_train_core.sh" "${1}" 1 1 "${2:-}"
