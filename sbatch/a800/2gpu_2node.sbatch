#!/bin/bash
# ============================================================
# A800 双卡双节点训练 (2 GPU, 2 节点, 每节点 24 核 CPU)
# ============================================================
# 用法 (推荐): sbatch sbatch/a800/2gpu_2node.sbatch "+experiment=exp001_baseline"
# 用法 (旧配置): sbatch sbatch/a800/2gpu_2node.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p nvlink
#SBATCH --qos=nvlinkg8
#SBATCH --job-name=train
#SBATCH --nodes=2
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
export POCKET_PLUS_LOCAL_CACHE_ENABLED="${POCKET_PLUS_LOCAL_CACHE_ENABLED:-1}"
export POCKET_PLUS_LOCAL_CACHE_ENV_PACK="${POCKET_PLUS_LOCAL_CACHE_ENV_PACK:-/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz}"
export POCKET_PLUS_LOCK_PRE_ENABLED="${POCKET_PLUS_LOCK_PRE_ENABLED:-0}"
export POCKET_PLUS_LOCK_AFTER_ENABLED="${POCKET_PLUS_LOCK_AFTER_ENABLED:-1}"
source "${SBATCH_ROOT}/_common.sh"
source "${SBATCH_ROOT}/_train_core.sh" "${1:-}" 2 1 "${2:-}"
