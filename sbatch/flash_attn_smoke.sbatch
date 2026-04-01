#!/bin/bash
# ============================================================
# 任意 GPU 节点上的 flash-attn / torch / PyG / spconv 冒烟测试
# ------------------------------------------------------------
# 默认在 H100 单卡节点上执行，可通过 sbatch 命令行覆盖分区/QOS/GRES：
#   sbatch sbatch/flash_attn_smoke.sbatch
#   sbatch -p a100 --qos=a100g2 --gres=gpu:a100:1 sbatch/flash_attn_smoke.sbatch
#
# 关键环境变量（可选）：
#   Pocket_Plus_CONDA_ENV_NAME=Pocket_Plus_centos7_cu121_allgpu
#   CUDA_HOME=/usr/local/cuda
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p h100
#SBATCH --qos=h100g2
#SBATCH --job-name=fa_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_ROOT="/home/penghongen/My_Project/Pocket_Plus"
CONDA_BASE="/home/penghongen/anaconda3"
CONDA_ENV_NAME="${Pocket_Plus_CONDA_ENV_NAME:-Pocket_Plus_centos7_cu121_allgpu}"
CUDA_HOME_DEFAULT="${CUDA_HOME:-/usr/local/cuda}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
unset BOOST_ROOT

export CUDA_HOME="${CUDA_HOME_DEFAULT}"
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
echo "Conda Env: ${CONDA_ENV_NAME}"
echo "Python: $(command -v python)"
echo "CUDA_HOME: ${CUDA_HOME}"
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

python - <<'PY'
import flash_attn
import lightning
import spconv.pytorch as spconv
import torch
import torch_cluster
import torch_geometric
import torch_scatter

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("flash_attn:", getattr(flash_attn, "__version__", "<unknown>"))
print("lightning:", lightning.__version__)
print("spconv:", getattr(spconv, "__version__", "<unknown>"))
print("torch_geometric:", torch_geometric.__version__)
print("torch_scatter:", getattr(torch_scatter, "__version__", "<unknown>"))
print("torch_cluster:", getattr(torch_cluster, "__version__", "<unknown>"))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this job.")

device = torch.device("cuda")
print("cuda.device_name:", torch.cuda.get_device_name(device))
print("cuda.device_capability:", torch.cuda.get_device_capability(device))

qkv = torch.randn(128, 3, 4, 64, device=device, dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
out = flash_attn.flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    64,
    0.0,
    causal=False,
)
torch.cuda.synchronize()
print("flash_attn_varlen_qkvpacked_func:", tuple(out.shape))
PY

echo "[Done] flash-attn smoke test passed."
