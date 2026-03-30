#!/bin/bash
# ============================================================
# H200 单卡训练 (1 GPU, 24 核 CPU)
# ============================================================
# 用法 (推荐):
#   sbatch sbatch/h200/1gpu.sbatch "+experiment=exp001_baseline"
# 用法 (旧配置):
#   sbatch sbatch/h200/1gpu.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# 示例:
#   sbatch sbatch/h200/1gpu.sbatch sparse_h100
#   sbatch sbatch/h200/1gpu.sbatch sparse_h100 "model/pseudo_atom=uniform_32 tag=test"
#   sbatch sbatch/h200/1gpu.sbatch sparse_h100 "+experiment=exp001_baseline"
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h200
#SBATCH --qos=h200g2
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
source "${SCRIPT_DIR}/../_train_core.sh" "${1}" 1 1 "${2:-}"
