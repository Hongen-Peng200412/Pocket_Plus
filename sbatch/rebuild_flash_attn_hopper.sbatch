#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h100
#SBATCH --qos=h100g2
#SBATCH --job-name=flashfix
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

PROJECT_ROOT="/home/penghongen/My_Project/Pocket_Plus"
CONDA_BASE="/home/penghongen/anaconda3"
CONDA_ENV_NAME="${Pocket_Plus_CONDA_ENV_NAME:-Pocket_Plus_centos7}"
STACK="${POCKET_TORCH_STACK:-torch24-cu121}"
MAX_JOBS="${MAX_JOBS:-8}"
POCKET_TORCH_CUDA_ARCH_LIST="${POCKET_TORCH_CUDA_ARCH_LIST:-9.0}"

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda-12.1" ]; then
        export CUDA_HOME="/usr/local/cuda-12.1"
    else
        export CUDA_HOME="/usr/local/cuda"
    fi
fi

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
unset BOOST_ROOT

export PATH="${CUDA_HOME}/bin:${PATH}"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64"
fi
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${CONDA_PREFIX}/lib/libgcc_s.so.1"

cd "${PROJECT_ROOT}" || { echo "Error: Could not cd to ${PROJECT_ROOT}"; exit 1; }

echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python)"
echo "Conda Env: ${CONDA_ENV_NAME}"
echo "STACK: ${STACK}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST: ${POCKET_TORCH_CUDA_ARCH_LIST}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "LD_PRELOAD: ${LD_PRELOAD}"
echo "glibc: $(ldd --version 2>/dev/null | head -n 1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc -V
fi

POCKET_FLASH_ONLY=1 \
POCKET_FLASH_REINSTALL=1 \
POCKET_TORCH_STACK="${STACK}" \
POCKET_TORCH_CUDA_ARCH_LIST="${POCKET_TORCH_CUDA_ARCH_LIST}" \
MAX_JOBS="${MAX_JOBS}" \
CUDA_HOME="${CUDA_HOME}" \
bash setup_env_centos7.sh environment_Pocket_Plus_centos7.yml "${CONDA_ENV_NAME}"

echo "[Done] flash-attn rebuild finished."
