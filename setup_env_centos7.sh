#!/bin/bash

set -euo pipefail

ENV_FILE="${1:-environment_Pocket_Plus_centos7.yml}"
ENV_NAME="${2:-Pocket_Plus_centos7}"
STACK="${POCKET_TORCH_STACK:-torch24-cu121}"
MAX_JOBS="${MAX_JOBS:-4}"
FLASH_ONLY="${POCKET_FLASH_ONLY:-0}"
FLASH_REINSTALL="${POCKET_FLASH_REINSTALL:-1}"
POCKET_TORCH_CUDA_ARCH_LIST="${POCKET_TORCH_CUDA_ARCH_LIST:-9.0}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1200}"
PIP_RETRIES="${PIP_RETRIES:-8}"
PIP_INSTALL_RETRY_ROUNDS="${PIP_INSTALL_RETRY_ROUNDS:-3}"
POCKET_UPGRADE_PIP="${POCKET_UPGRADE_PIP:-0}"
CUDA_HOME_DEFAULT="${CUDA_HOME:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[Env] conda was not found in PATH."
    exit 1
fi

sanitize_inherited_build_env() {
    local vars=(CC CXX CPP CUDAHOSTCXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDSHARED AR AS NM RANLIB STRIP OBJCOPY OBJDUMP CONDA_BUILD_SYSROOT)
    local var
    for var in "${vars[@]}"; do
        if [ -n "${!var:-}" ]; then
            echo "[Env] Clearing inherited ${var}=${!var}"
            unset "${var}"
        fi
    done
}

echo "[Env] Sanitizing inherited compiler/build environment before conda env create"
sanitize_inherited_build_env

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u

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

if [ "${CONDA_DEFAULT_ENV:-}" = "${ENV_NAME}" ]; then
    echo "[Env] '${ENV_NAME}' is already active, skipping redundant conda activate."
else
    set +u
    conda activate "${ENV_NAME}"
    set -u
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT

configure_stack() {
    local selected_stack="$1"
    case "${selected_stack}" in
        torch24-cu121)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
            TORCH_SPEC="torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
            PYG_WHL_URL="https://data.pyg.org/whl/torch-2.4.0+cu121.html"
            SPCONV_SPEC="spconv-cu121==2.3.8"
            FLASH_ATTN_SPEC="flash-attn==2.8.3"
            MIN_CUDA_VERSION="12.0"
            EXPECTED_TORCH_CUDA="12.1"
            CUDA_HOME_CANDIDATES=("/usr/local/cuda-12.1" "/usr/local/cuda")
            ;;
        torch21-cu118)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
            TORCH_SPEC="torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2"
            PYG_WHL_URL="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
            SPCONV_SPEC="spconv-cu118==2.3.8"
            FLASH_ATTN_SPEC="flash-attn==2.5.8"
            MIN_CUDA_VERSION="11.6"
            EXPECTED_TORCH_CUDA="11.8"
            CUDA_HOME_CANDIDATES=("/usr/local/cuda-11.8" "/usr/local/cuda")
            ;;
        *)
            echo "[Env] Unknown POCKET_TORCH_STACK='${selected_stack}'. Supported values: torch24-cu121, torch21-cu118"
            exit 1
            ;;
    esac
}

detect_installed_torch() {
    python - <<'PY'
try:
    import torch
except Exception:
    print("NONE|")
else:
    version = getattr(torch, "__version__", "UNKNOWN")
    cuda_version = getattr(torch.version, "cuda", None) or ""
    print(f"{version}|{cuda_version}")
PY
}

write_runtime_hooks() {
    local activate_dir deactivate_dir
    activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
    deactivate_dir="${CONDA_PREFIX}/etc/conda/deactivate.d"
    mkdir -p "${activate_dir}" "${deactivate_dir}"

    cat > "${activate_dir}/pocket_plus_runtime.sh" <<EOF
export _POCKET_PLUS_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export _POCKET_PLUS_OLD_LD_PRELOAD="\${LD_PRELOAD:-}"
export CUDA_HOME="${CUDA_HOME_DEFAULT}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="/usr/lib64:\${CUDA_HOME}/lib64"
if [ -n "\${_POCKET_PLUS_OLD_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:\${_POCKET_PLUS_OLD_LD_LIBRARY_PATH}"
fi
export LD_PRELOAD="\${CONDA_PREFIX}/lib/libstdc++.so.6:\${CONDA_PREFIX}/lib/libgcc_s.so.1"
EOF

    cat > "${deactivate_dir}/pocket_plus_runtime.sh" <<'EOF'
export LD_LIBRARY_PATH="${_POCKET_PLUS_OLD_LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${_POCKET_PLUS_OLD_LD_PRELOAD:-}"
unset _POCKET_PLUS_OLD_LD_LIBRARY_PATH
unset _POCKET_PLUS_OLD_LD_PRELOAD
EOF
}

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

echo "[Env] Installing repo-side pure Python helpers"
pip_install_retry "rootutils==1.0.7"

declare -a CUDA_HOME_CANDIDATES=()
configure_stack "${STACK}"

INSTALLED_TORCH_INFO="$(detect_installed_torch)"
INSTALLED_TORCH_VERSION="${INSTALLED_TORCH_INFO%%|*}"
INSTALLED_TORCH_CUDA="${INSTALLED_TORCH_INFO#*|}"
if [ "${INSTALLED_TORCH_VERSION}" != "NONE" ]; then
    echo "[Env] Installed torch: ${INSTALLED_TORCH_VERSION} (cuda=${INSTALLED_TORCH_CUDA:-cpu})"
fi

if [ "${FLASH_ONLY}" = "1" ] && [ "${INSTALLED_TORCH_VERSION}" != "NONE" ]; then
    case "${INSTALLED_TORCH_CUDA}" in
        11.8)
            if [ "${STACK}" != "torch21-cu118" ]; then
                echo "[Env] FLASH_ONLY detected torch cuda=${INSTALLED_TORCH_CUDA}, overriding STACK -> torch21-cu118"
                STACK="torch21-cu118"
                configure_stack "${STACK}"
            fi
            ;;
        12.1|12.2)
            if [ "${STACK}" != "torch24-cu121" ]; then
                echo "[Env] FLASH_ONLY detected torch cuda=${INSTALLED_TORCH_CUDA}, overriding STACK -> torch24-cu121"
                STACK="torch24-cu121"
                configure_stack "${STACK}"
            fi
            ;;
        "")
            echo "[Env] FLASH_ONLY requested but installed torch reports no CUDA runtime."
            exit 1
            ;;
        *)
            echo "[Env] FLASH_ONLY does not know how to map torch cuda=${INSTALLED_TORCH_CUDA} to a supported stack."
            exit 1
            ;;
    esac
fi

if [ -z "${CUDA_HOME_DEFAULT}" ]; then
    for candidate in "${CUDA_HOME_CANDIDATES[@]}"; do
        if [ -d "${candidate}" ]; then
            CUDA_HOME_DEFAULT="${candidate}"
            break
        fi
    done
    CUDA_HOME_DEFAULT="${CUDA_HOME_DEFAULT:-/usr/local/cuda}"
fi
echo "[Env] Using CUDA_HOME default: ${CUDA_HOME_DEFAULT}"

if [ "${FLASH_ONLY}" != "1" ]; then
    echo "[Env] Installing PyTorch stack: ${STACK}"
    pip_install_retry --index-url "${TORCH_INDEX_URL}" --force-reinstall --no-cache-dir ${TORCH_SPEC}

    echo "[Env] Verifying that torch now sees a CUDA runtime"
    python - <<'PY'
import torch
cuda_version = getattr(torch.version, "cuda", None)
print("torch:", torch.__version__)
print("torch.version.cuda:", cuda_version)
if not cuda_version:
    raise SystemExit(
        "[Env] torch still reports a CPU-only runtime after CUDA wheel installation. "
        "Check whether a conda package pulled in cpu-only pytorch/torchvision."
    )
PY

    echo "[Env] Installing torch-dependent training packages"
    pip_install_retry \
        "lightning==2.2.5" \
        "pytorch-lightning==2.2.5" \
        "torchmetrics==1.3.2" \
        "timm==0.9.16"

    echo "[Env] Installing PyG runtime packages"
    pip_install_retry \
        "torch-scatter==2.1.2" \
        "torch-cluster==1.6.3" \
        -f "${PYG_WHL_URL}"
    pip_install_retry "torch-geometric==2.6.1"

    echo "[Env] Installing spconv"
    pip_install_retry "${SPCONV_SPEC}"
else
    echo "[Env] POCKET_FLASH_ONLY=1, skipping torch / PyG / spconv reinstall."
fi

echo "[Env] Writing Pocket_Plus runtime activation hooks"
write_runtime_hooks
source "${CONDA_PREFIX}/etc/conda/activate.d/pocket_plus_runtime.sh"

echo "[Env] Verifying compiler toolchain for flash-attn"
if [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
elif [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" ]; then
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
else
    echo "[Env] Conda GCC toolchain was not found under ${CONDA_PREFIX}/bin."
    exit 1
fi
export CUDAHOSTCXX="${CXX}"
echo "[Env] CC=${CC}"
echo "[Env] CXX=${CXX}"
echo "[Env] CUDAHOSTCXX=${CUDAHOSTCXX}"
${CC} --version | head -n 1
${CXX} --version | head -n 1

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

if [ -n "${INSTALLED_TORCH_CUDA:-}" ]; then
    python - <<PY
from packaging.version import Version

installed = Version("${INSTALLED_TORCH_CUDA}")
nvcc = Version("${NVCC_VERSION}")
stack = "${STACK}"
expected = Version("${EXPECTED_TORCH_CUDA}")

if stack == "torch21-cu118" and nvcc.release[:2] != installed.release[:2]:
    raise SystemExit(
        f"[Env] nvcc {nvcc} mismatches installed torch cuda {installed}. "
        f"Set CUDA_HOME to a CUDA {installed} toolkit (for example /usr/local/cuda-11.8) "
        f"or reinstall the full torch stack."
    )

if stack == "torch24-cu121" and nvcc.release[0] != expected.release[0]:
    raise SystemExit(
        f"[Env] nvcc {nvcc} is incompatible with stack {stack}. Expected CUDA major {expected.release[0]}."
    )
print(f"[Env] nvcc {nvcc} is acceptable for installed torch cuda {installed} under stack {stack}.")
PY
fi

echo "[Env] Installing flash-attn from source for glibc 2.17 safety"
export TORCH_CUDA_ARCH_LIST="${POCKET_TORCH_CUDA_ARCH_LIST}"
echo "[Env] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
if [ "${FLASH_REINSTALL}" = "1" ]; then
    echo "[Env] Removing existing flash-attn wheel before rebuild"
    python -m pip uninstall -y flash-attn flash_attn || true
fi
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    CC="${CC}" CXX="${CXX}" CUDAHOSTCXX="${CUDAHOSTCXX}" \
    pip_install_retry "${FLASH_ATTN_SPEC}" --no-build-isolation --no-cache-dir --force-reinstall --no-deps

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
if torch.cuda.is_available():
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

echo "[Env] Done."
echo "[Env] If flash-attn build time is too long, retry with a larger MAX_JOBS."
echo "[Env] If torch24-cu121 fails because the host only has CUDA 11.x toolkit, retry with:"
echo "[Env]   POCKET_TORCH_STACK=torch21-cu118 bash setup_env_centos7.sh ${ENV_FILE} ${ENV_NAME}"
