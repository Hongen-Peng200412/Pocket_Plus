#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h100
#SBATCH --qos=h100g2
#SBATCH --job-name=pp_debug_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24

set -euo pipefail

source /home/penghongen/anaconda3/bin/activate Pocket_Plus

export PROJECT_ROOT="/home/penghongen/My_Project"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Pocket_Plus:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Job $SLURM_JOB_ID allocated on nodes: $SLURM_JOB_NODELIST"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
echo "glibc: $(ldd --version 2>/dev/null | head -n 1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"

echo "[Check] Import torch only"
python - <<'PY'
import os
import sys

print("sys.version:", sys.version.replace("\n", " "))
print("sys.executable:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

echo "[Run] Start single-process training without SLURM network helper"
python Pocket_Plus/src/train.py \
    --config=sparse_h100 \
    train.nnodes=1 \
    train.devices=1
