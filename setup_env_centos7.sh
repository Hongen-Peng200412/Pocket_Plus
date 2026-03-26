#!/bin/bash

set -euo pipefail

ENV_FILE="${1:-environment_Pocket_Plus_centos7.yml}"
ENV_NAME="${2:-Pocket_Plus_centos7}"
STACK="${POCKET_TORCH_STACK:-torch24-cu121}"
MAX_JOBS="${MAX_JOBS:-4}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1200}"
PIP_RETRIES="${PIP_RETRIES:-8}"
PIP_INSTALL_RETRY_ROUNDS="${PIP_INSTALL_RETRY_ROUNDS:-3}"
POCKET_UPGRADE_PIP="${POCKET_UPGRADE_PIP:-0}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[Env] conda was not found in PATH."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[Env] Creating base environment '${ENV_NAME}' from ${ENV_FILE}"
    if ! conda env create -f "${ENV_FILE}" -n "${ENV_NAME}"; then
        echo "[Env] Default solver failed while creating '${ENV_NAME}'."
        echo "[Env] Retrying with the classic solver..."
        conda env create --solver classic -f "${ENV_FILE}" -n "${ENV_NAME}"
    fi
else
    echo "[Env] Reusing existing environment '${ENV_NAME}'"
fi

conda activate "${ENV_NAME}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT

pip_install_retry() {
    local round=1
    while [ "${round}" -le "${PIP_INSTALL_RETRY_ROUNDS}" ]; do
        if python -m pip install \
            --timeout "${PIP_DEFAULT_TIMEOUT}" \
            --retries "${PIP_RETRIES}" \
            "$@"; then
            return 0
        fi
        echo "[Env] pip install failed (round ${round}/${PIP_INSTALL_RETRY_ROUNDS}): $*"
        if [ "${round}" -eq "${PIP_INSTALL_RETRY_ROUNDS}" ]; then
            return 1
        fi
        round=$((round + 1))
        sleep 10
    done
}

echo "[Env] Current pip: $(python -m pip --version)"
if [ "${POCKET_UPGRADE_PIP}" = "1" ]; then
    echo "[Env] Upgrading pip because POCKET_UPGRADE_PIP=1"
    pip_install_retry --upgrade pip
else
    echo "[Env] Skipping pip self-upgrade (set POCKET_UPGRADE_PIP=1 to enable)"
fi

case "${STACK}" in
    torch24-cu121)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
        TORCH_SPEC="torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
        PYG_WHL_URL="https://data.pyg.org/whl/torch-2.4.0+cu121.html"
        SPCONV_SPEC="spconv-cu121==2.3.8"
        FLASH_ATTN_SPEC="flash-attn==2.8.3"
        MIN_CUDA_VERSION="12.0"
        ;;
    torch21-cu118)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        TORCH_SPEC="torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2"
        PYG_WHL_URL="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
        SPCONV_SPEC="spconv-cu118==2.3.8"
        FLASH_ATTN_SPEC="flash-attn==2.5.8"
        MIN_CUDA_VERSION="11.6"
        ;;
    *)
        echo "[Env] Unknown POCKET_TORCH_STACK='${STACK}'. Supported values: torch24-cu121, torch21-cu118"
        exit 1
        ;;
esac

echo "[Env] Installing PyTorch stack: ${STACK}"
pip_install_retry --index-url "${TORCH_INDEX_URL}" ${TORCH_SPEC}

echo "[Env] Installing PyG runtime packages"
pip_install_retry \
    "torch-scatter==2.1.2" \
    "torch-cluster==1.6.3" \
    -f "${PYG_WHL_URL}"
pip_install_retry "torch-geometric==2.6.1"

echo "[Env] Installing spconv"
pip_install_retry "${SPCONV_SPEC}"

if [ -z "${CUDA_HOME:-}" ] && [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[Env] nvcc was not found. flash-attn source build requires a CUDA toolkit."
    echo "[Env] Set CUDA_HOME to a toolkit >= ${MIN_CUDA_VERSION}, or stop here and install flash-attn separately."
    exit 1
fi

NVCC_VERSION="$(nvcc -V | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [ -z "${NVCC_VERSION}" ]; then
    echo "[Env] Failed to parse nvcc version."
    exit 1
fi

python - <<PY
from packaging.version import Version

nvcc = Version("${NVCC_VERSION}")
minimum = Version("${MIN_CUDA_VERSION}")
if nvcc < minimum:
    raise SystemExit(
        f"[Env] nvcc {nvcc} is too old for ${STACK}. Need CUDA toolkit >= {minimum}."
    )
print(f"[Env] nvcc {nvcc} satisfies minimum CUDA toolkit requirement {minimum}.")
PY

echo "[Env] Installing flash-attn from source for glibc 2.17 safety"
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" \
    pip_install_retry "${FLASH_ATTN_SPEC}" --no-build-isolation

echo "[Env] Running import smoke test"
python - <<'PY'
import flash_attn
import spconv.pytorch as spconv
import torch
import torch_cluster
import torch_geometric
import torch_scatter

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("flash_attn:", getattr(flash_attn, "__version__", "<unknown>"))
print("spconv:", getattr(spconv, "__version__", "<unknown>"))
print("torch_geometric:", torch_geometric.__version__)
print("torch_scatter:", getattr(torch_scatter, "__version__", "<unknown>"))
print("torch_cluster:", getattr(torch_cluster, "__version__", "<unknown>"))
PY

echo "[Env] Done."
echo "[Env] If flash-attn build time is too long, retry with a larger MAX_JOBS."
echo "[Env] If torch24-cu121 fails because the host only has CUDA 11.x toolkit, retry with:"
echo "[Env]   POCKET_TORCH_STACK=torch21-cu118 bash setup_env_centos7.sh ${ENV_FILE} ${ENV_NAME}"
