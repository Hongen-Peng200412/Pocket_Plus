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

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
export POCKET_PLUS_LOCK_PRE_ENABLED="${POCKET_PLUS_LOCK_PRE_ENABLED:0}"
export POCKET_PLUS_LOCK_AFTER_ENABLED="${POCKET_PLUS_LOCK_AFTER_ENABLED:-1}"
source "${SBATCH_ROOT}/_common.sh"
source "${SBATCH_ROOT}/_train_core.sh" "${1}" 1 1 "${2:-}"
