#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p nvlink
#SBATCH --qos=nvlinkg8
#SBATCH --job-name=non
#SBATCH --nodes=1
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

CONFIG_NAME="non_a800"
PROJECT_ROOT="/home/penghongen/My_Project"
CONDA_BASE="/home/penghongen/anaconda3"
CONDA_ENV_NAME="${Pocket_Plus_CONDA_ENV_NAME:-Pocket_Plus_centos7}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
unset BOOST_ROOT

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

LOCK_PRE="/home/penghongen/pre_lock_${SLURM_JOB_ID}"
# touch "${LOCK_PRE}"
echo "[Status] pre_Lock created: ${LOCK_PRE}"
echo "[Wait] Waiting for user to delete ${LOCK_PRE} to start..."

LOCK_AFTER="/home/penghongen/after_lock_${SLURM_JOB_ID}"
touch "${LOCK_AFTER}"
echo "[Status] after_Lock created: ${LOCK_AFTER}"

LOCK_TRY="/home/penghongen/try_lock_${SLURM_JOB_ID}"

while [ -f "${LOCK_PRE}" ]; do
    sleep 20
done
echo "[Start] Starting Training..."

while true; do
    RUN_STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
    echo "[Run] Local run stamp: ${RUN_STAMP}"
    echo "================================================================================="
    echo "[Attempt] Starting Pocket_Plus training (${CONFIG_NAME})..."
    echo "================================================================================="

    if python Pocket_Plus/src/train.py \
        --config="${CONFIG_NAME}" \
        train.nnodes=1 \
        train.devices=1; then
        echo "================================================================================="
        echo "[Success] Training finished successfully."
        echo "================================================================================="
        rm -f "${LOCK_AFTER}"
        rm -f "${LOCK_TRY}"
        break
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[Error] Training FAILED! Check error logs above."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        touch "${LOCK_TRY}"

        echo "[Action Required] Job is PAUSED. "
        echo "   -> To RETRY: Delete file '${LOCK_TRY}'"
        echo "   -> To CANCEL: Delete file '${LOCK_AFTER}'"

        while [ -f "${LOCK_TRY}" ] && [ -f "${LOCK_AFTER}" ]; do
            sleep 20
        done

        if [ ! -f "${LOCK_AFTER}" ]; then
            echo "[Stop] '${LOCK_AFTER}' was deleted by user. Exiting loop."
            break
        else
            echo "[Retry] '${LOCK_TRY}' was deleted by user. Restarting task immediately..."
            continue
        fi
    fi
done

while [ -f "${LOCK_AFTER}" ]; do
    sleep 20
done

echo "[Exit] Job finished."
