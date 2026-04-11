#!/bin/bash
# ============================================================
# 训练/执行核心逻辑 (被各 GPU sbatch 调用)
# 特性 1: 动态挂载运行指令。在 try_lock 时可直接修改 ~/run_cmd_xxx.sh 来无缝切换参数甚至脚本。
# 特性 2: 支持通过配置关闭锁机制，例如: export POCKET_PLUS_LOCK_AFTER_ENABLED=0，0 和 -0 均表示关闭。
# 特性 3: 支持 TRY_AFTER_END 机制，在任务结束后默认挂起不退出节点，以便连续执行相关任务。
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
TRY_AFTER_END_ENABLED="${POCKET_PLUS_TRY_AFTER_END_ENABLED:-1}"

echo "[Args] LOCK_PRE_ENABLED=${LOCK_PRE_ENABLED}"
echo "[Args] LOCK_AFTER_ENABLED=${LOCK_AFTER_ENABLED}"
echo "[Args] TRY_AFTER_END_ENABLED=${TRY_AFTER_END_ENABLED}"

# ============================================================
# 信号捕获: 确保所有退出路径都清理临时文件
# ============================================================
cleanup() {
    rm -f "/home/penghongen/run_cmd_${SLURM_JOB_ID}.sh"
    rm -f "/home/penghongen/pre_lock_${SLURM_JOB_ID}"
    rm -f "/home/penghongen/after_lock_${SLURM_JOB_ID}"
    rm -f "/home/penghongen/try_lock_${SLURM_JOB_ID}"
    echo "[Cleanup] Removed lock/cmd files for job ${SLURM_JOB_ID}."
}
trap cleanup EXIT SIGTERM SIGINT










RUN_CMD_FILE="/home/penghongen/run_cmd_${SLURM_JOB_ID}.sh"
# ============================================================
# 提前生成指令挂载文件
# 若 sbatch 脚本中定义了 write_custom_run_cmd_hook 函数，则优先调用它
# ============================================================
if [ ! -f "${RUN_CMD_FILE}" ]; then
    if type write_custom_run_cmd_hook >/dev/null 2>&1; then
        echo "[Info] 'write_custom_run_cmd_hook' found, using it to generate RUN_CMD_FILE."
        write_custom_run_cmd_hook "${RUN_CMD_FILE}"
    else
        cat << EOF > "${RUN_CMD_FILE}"
#!/bin/bash
# ============================================================
# Dynamic Command Wrapper for Job ${SLURM_JOB_ID}
# NOTE: You can dynamically edit this file while the job is paused 
# via 'try_lock'. The next retry loop will execute your changes!
# ============================================================

python Pocket_Plus/src/train.py \\
    --config="${CONFIG_NAME}" \\
    ${HYDRA_OVERRIDES} \\
    train.nnodes=${NNODES} \\
    train.devices=${DEVICES}
EOF
    fi
    chmod +x "${RUN_CMD_FILE}" 2>/dev/null || true
fi









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
    echo "[Attempt] Executing dynamic command from ${RUN_CMD_FILE}..."
    cat "${RUN_CMD_FILE}"
    echo "================================================================================="

    if bash "${RUN_CMD_FILE}"; then
        echo "================================================================================="
        echo "[Success] Execution finished successfully."
        echo "================================================================================="
        
        if is_lock_enabled "${TRY_AFTER_END_ENABLED}"; then
            echo "[Action Required] Task SUCCESS! But TRY_AFTER_END_ENABLED is enabled."
            echo "   -> Job will be PAUSED to keep the node allocation alive."
            echo "   -> To RUN ANOTHER TASK (e.g., Inference): Edit '${RUN_CMD_FILE}' then delete file '${LOCK_TRY}'"
            echo "   -> To END JOB & RELEASE NODE: Delete file '${LOCK_AFTER}'"
            
            touch "${LOCK_TRY}"
            while [ -f "${LOCK_TRY}" ] && [ -f "${LOCK_AFTER}" ]; do
                sleep 20
            done

            if [ ! -f "${LOCK_AFTER}" ]; then
                echo "[Stop] '${LOCK_AFTER}' was deleted by user. Exiting loop."
                break
            else
                echo "[Retry] '${LOCK_TRY}' was deleted by user. Starting new task..."
                continue
            fi
        else
            rm -f "${LOCK_AFTER}"
            rm -f "${LOCK_TRY}"
            break
        fi
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[Error] Execution FAILED! Check error logs above."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        if ! is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
            echo "[Stop] after_Lock is disabled, so the job will exit immediately after this failure."
            break
        fi

        touch "${LOCK_TRY}"

        echo "[Action Required] Job is PAUSED. "
        echo "   -> To EDIT PARAMETERS or SCRIPT: Modify '${RUN_CMD_FILE}'"
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

# cleanup() 已由 EXIT trap 自动调用, 无需手动 rm
echo "[Exit] Job finished."
