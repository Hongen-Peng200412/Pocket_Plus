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

CONFIG_NAME="${1:?ERROR: 必须提供 CONFIG_NAME 作为第一个参数}"
NNODES="${2:?ERROR: 必须提供 NNODES 作为第二个参数}"
DEVICES="${3:?ERROR: 必须提供 DEVICES 作为第三个参数}"
HYDRA_OVERRIDES="${4:-}"

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
