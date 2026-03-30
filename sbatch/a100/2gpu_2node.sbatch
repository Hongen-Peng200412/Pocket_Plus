#!/bin/bash
# ============================================================
# A100 双卡双节点训练 (2 GPU, 2 节点, 每节点 16 核 CPU)
# ============================================================
# 用法 (推荐): sbatch sbatch/a100/2gpu_2node.sbatch "+experiment=exp001_baseline"
# 用法 (旧配置): sbatch sbatch/a100/2gpu_2node.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH --job-name=train
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
source "${SCRIPT_DIR}/../_train_core.sh" "${1}" 2 1 "${2:-}"
