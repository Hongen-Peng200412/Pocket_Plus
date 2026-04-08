#!/bin/bash
# ============================================================
# 公共环境设置脚本
# ============================================================
# 所有 sbatch 脚本在训练前 source 此文件
# 用法: source "$(dirname "$0")/../_common.sh"
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/penghongen/My_Project"
CONDA_BASE="/home/penghongen/anaconda3"
CONDA_ENV_NAME="${Pocket_Plus_CONDA_ENV_NAME:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
unset BOOST_ROOT

# --- Locale (防止中文路径乱码) ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64"
fi
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${CONDA_PREFIX}/lib/libgcc_s.so.1"

export PROJECT_ROOT
cd "${PROJECT_ROOT}" || { echo "Error: Could not cd to ${PROJECT_ROOT}"; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Pocket_Plus:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Job ${SLURM_JOB_ID} allocated on nodes: ${SLURM_JOB_NODELIST}"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python)"
echo "Conda Env: ${CONDA_ENV_NAME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "LD_PRELOAD: ${LD_PRELOAD}"
echo "glibc: $(ldd --version 2>/dev/null | head -n 1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"
