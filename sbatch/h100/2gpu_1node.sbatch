#!/bin/bash
# ============================================================
# H100 双卡单节点训练 (2 GPU, 48 核 CPU)
# ============================================================
# 用法 (推荐): sbatch sbatch/h100/2gpu_1node.sbatch "+experiment=exp001_baseline"
# 用法 (旧配置): sbatch sbatch/h100/2gpu_1node.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h100
#SBATCH --qos=h100g2
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
source "${SCRIPT_DIR}/../_train_core.sh" "${1}" 1 2 "${2:-}"
