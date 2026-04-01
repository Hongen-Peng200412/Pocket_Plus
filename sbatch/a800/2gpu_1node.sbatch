#!/bin/bash
# ============================================================
# A800 双卡单节点训练 (2 GPU, 48 核 CPU)
# ============================================================
# 用法 (推荐): sbatch sbatch/a800/2gpu_1node.sbatch "+experiment=exp001_baseline"
# 用法 (旧配置): sbatch sbatch/a800/2gpu_1node.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p a800
#SBATCH --qos=a800g2
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:a800:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
export POCKET_PLUS_LOCK_PRE_ENABLED="${POCKET_PLUS_LOCK_PRE_ENABLED:-1}"
export POCKET_PLUS_LOCK_AFTER_ENABLED="${POCKET_PLUS_LOCK_AFTER_ENABLED:-1}"
source "${SBATCH_ROOT}/_common.sh"
source "${SBATCH_ROOT}/_train_core.sh" "${1}" 1 2 "${2:-}"
