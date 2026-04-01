#!/bin/bash
# ============================================================
# 新建独立的 cu121 全 GPU 环境
# ------------------------------------------------------------
# 默认在 H100 单卡节点上构建，可通过 sbatch 命令行覆盖分区/QOS/GRES：
#   sbatch sbatch/install_env_cu121_allgpu.sbatch
#   sbatch -p h200 --qos=h200g2 --gres=gpu:h200:1 sbatch/install_env_cu121_allgpu.sbatch
#
# 关键环境变量（可选）：
#   Pocket_Plus_CONDA_ENV_NAME=Pocket_Plus_centos7_cu121_allgpu
#   POCKET_TORCH_STACK=torch24-cu121
#   POCKET_TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0'
#   CUDA_HOME=/usr/local/cuda
#   MAX_JOBS=8
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h100
#SBATCH --qos=h100g2
#SBATCH --job-name=env_cu121
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

PROJECT_ROOT="/home/penghongen/My_Project/Pocket_Plus"
CONDA_BASE="/home/penghongen/anaconda3"
CONDA_ENV_NAME="${Pocket_Plus_CONDA_ENV_NAME:-Pocket_Plus_centos7_cu121_allgpu}"
STACK="${POCKET_TORCH_STACK:-torch24-cu121}"
TORCH_ARCH_LIST="${POCKET_TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_HOME_DEFAULT="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS="${MAX_JOBS:-8}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u
unset BOOST_ROOT

export CUDA_HOME="${CUDA_HOME_DEFAULT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64"
fi

cd "${PROJECT_ROOT}" || { echo "Error: Could not cd to ${PROJECT_ROOT}"; exit 1; }

echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Target Env: ${CONDA_ENV_NAME}"
echo "STACK: ${STACK}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_ARCH_LIST}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "glibc: $(ldd --version 2>/dev/null | head -n 1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc -V
fi

POCKET_TORCH_STACK="${STACK}" \
POCKET_TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" \
MAX_JOBS="${MAX_JOBS}" \
CUDA_HOME="${CUDA_HOME}" \
bash setup_env_centos7.sh environment_Pocket_Plus_centos7.yml "${CONDA_ENV_NAME}"

echo "[Done] Environment build finished: ${CONDA_ENV_NAME}"
