#!/bin/bash
# ============================================================
# 训练核心逻辑 (被各 GPU sbatch 调用)
# ============================================================
# 用法: _train_core.sh CONFIG_NAME NNODES DEVICES [HYDRA_OVERRIDES]
# 参数:
#   $1 = CONFIG_NAME (必须), Hydra 配置名, 如 "sparse_h100"
#   $2 = NNODES (必须), 节点数
#   $3 = DEVICES (必须), 每节点 GPU 数
#   $4 = HYDRA_OVERRIDES (可选), Hydra 覆盖参数, 空格分隔
# ============================================================

# 智能参数解析: 若 $1 以 '+' 开头, 视为 HYDRA_OVERRIDES, CONFIG_NAME 默认 "base"
if [[ "${1:-}" == +* ]]; then
    CONFIG_NAME="base"
    HYDRA_OVERRIDES="${1:-}"
else
    CONFIG_NAME="${1:-base}"
    HYDRA_OVERRIDES="${4:-}"
fi

NNODES="${2:?Error: NNODES is required}"
DEVICES="${3:?Error: DEVICES is required}"

echo "[Args] CONFIG_NAME=${CONFIG_NAME}"
echo "[Args] HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-<empty>}"
echo "[Args] NNODES=${NNODES}"
echo "[Args] DEVICES=${DEVICES}"

is_lock_enabled() {
    case "${1:-1}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

LOCK_PRE_ENABLED="${POCKET_PLUS_LOCK_PRE_ENABLED:-1}"
LOCK_AFTER_ENABLED="${POCKET_PLUS_LOCK_AFTER_ENABLED:-1}"

echo "[Args] LOCK_PRE_ENABLED=${LOCK_PRE_ENABLED}"
echo "[Args] LOCK_AFTER_ENABLED=${LOCK_AFTER_ENABLED}"

LOCK_PRE="/home/penghongen/pre_lock_${SLURM_JOB_ID}"
if is_lock_enabled "${LOCK_PRE_ENABLED}"; then
    touch "${LOCK_PRE}"
    echo "[Status] pre_Lock created: ${LOCK_PRE}"
    echo "[Wait] Waiting for user to delete ${LOCK_PRE} to start..."
else
    echo "[Status] pre_Lock disabled by sbatch script."
fi

LOCK_AFTER="/home/penghongen/after_lock_${SLURM_JOB_ID}"
if is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
    touch "${LOCK_AFTER}"
    echo "[Status] after_Lock created: ${LOCK_AFTER}"
else
    echo "[Status] after_Lock disabled by sbatch script."
fi

LOCK_TRY="/home/penghongen/try_lock_${SLURM_JOB_ID}"

if is_lock_enabled "${LOCK_PRE_ENABLED}"; then
    while [ -f "${LOCK_PRE}" ]; do
        sleep 20
    done
fi
echo "[Start] Starting Training..."

while true; do
    RUN_STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
    echo "[Run] Local run stamp: ${RUN_STAMP}"
    echo "================================================================================="
    echo "[Attempt] Starting Pocket_Plus training (${CONFIG_NAME})..."
    echo "================================================================================="

    if python Pocket_Plus/src/train.py \
        --config="${CONFIG_NAME}" \
        ${HYDRA_OVERRIDES} \
        train.nnodes=${NNODES} \
        train.devices=${DEVICES}; then
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

        if ! is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
            echo "[Stop] after_Lock is disabled, so the job will exit immediately after this failure."
            break
        fi

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

if is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
    while [ -f "${LOCK_AFTER}" ]; do
        sleep 20
    done
fi

echo "[Exit] Job finished."
