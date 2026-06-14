#!/bin/bash
# ============================================================
# 训练/执行核心逻辑 (被各 GPU sbatch 调用)
# ============================================================
# 用法: _train_core.sh CONFIG_NAME NNODES DEVICES [HYDRA_OVERRIDES]
# 参数:
#   $1 = CONFIG_NAME (必须), Hydra 配置名, 如 "sparse_h100"
#   $2 = NNODES (必须), 节点数
#   $3 = DEVICES (必须), 每节点 GPU 数
#   $4 = HYDRA_OVERRIDES (可选), Hydra 覆盖参数, 空格分隔
#
# ============================================================
# 功能列表:
#
# 特性 1 — 动态挂载运行指令:
#   在 try_lock 时可直接修改 ~/run_cmd_xxx.sh 来无缝切换参数甚至脚本。
#
# 特性 2 — 支持通过配置关闭锁机制:
#   例如: export POCKET_PLUS_LOCK_AFTER_ENABLED=0，0 和 -0 均表示关闭。
#
# 特性 3 — TRY_AFTER_END 机制:
#   在任务成功结束后默认挂起不退出节点，以便连续执行相关任务。
#
# 特性 4 — kill_lock 文件信号哨兵 (NFS 穿透杀进程):
#   当 Python 进程吃满计算节点导致 SSH / srun 都无法穿透时,
#   用户只需在 master 节点执行:
#     touch /home/penghongen/kill_lock_<JOBID>
#   后台哨兵每 10 秒检查一次该文件, 发现后立即 kill -9 整个进程组,
#   _train_core.sh 自动进入 try_lock 挂起, 从而保住节点 allocation。
#   使用示例:
#     touch /home/penghongen/kill_lock_234061   # 触发杀进程
#     # 等 ≤10 秒, .out 日志出现 [KillWatcher], 随后 try_lock 生成
#     vim /home/penghongen/run_cmd_234061.sh    # 修改下次要运行的命令
#     rm /home/penghongen/try_lock_234061       # 启动下一轮执行
#
# 特性 5 — 强制行缓冲输出:
#   使用 stdbuf -oL -eL 包裹 run_cmd 子进程, 确保 Python 的 stdout/stderr
#   在 Slurm 的非交互环境下也能逐行实时刷到 .out/.err 日志中,
#   彻底解决"日志看起来卡住、实际在运行"的问题。
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
# [特性 4] kill_lock 文件信号哨兵
# ============================================================
# 原理: 利用 NFS 共享文件系统, 在 master 上 touch 一个文件即可
# 穿透任何资源耗尽的计算节点, 杀掉 run_cmd 的整个进程树。
# 哨兵以后台子进程运行, 每 10 秒检查一次, 几乎不消耗 CPU/内存。
# ============================================================

# str, kill_lock 信号文件路径 (NFS 共享, 所有节点均可访问)
KILL_LOCK="/home/penghongen/kill_lock_${SLURM_JOB_ID}"
# int | "", 哨兵后台进程的 PID, 用于在 run_cmd 结束后清理哨兵
_KILL_WATCHER_PID=""
# int | "", 当前 run_cmd 进程的 PID (同时也是其进程组 ID, 因用 setsid 启动)
_RUN_CMD_PID=""

_start_kill_watcher() {
    # 启动后台哨兵子进程: 每 10 秒检查 KILL_LOCK 文件是否存在
    # 一旦检测到, 立即 kill -9 整个进程组 (bash + python 等所有子进程)
    #
    # 输入参数:
    #   $1: int, 被监控的 run_cmd 进程的 PID
    #       (因为用了 setsid 启动, 该 PID 同时也是进程组 ID)
    local CMD_PID="$1"
    (
        while true; do
            sleep 10
            if [ -f "${KILL_LOCK}" ]; then
                echo "[KillWatcher] 检测到 ${KILL_LOCK}, 正在终止进程组 ${CMD_PID}..."
                # kill -9 -PID: 负号代表发送信号给整个进程组 (含 bash 及其 python 子进程)
                # 若进程组已不存在则回退到单进程 kill, 再失败则忽略
                kill -9 -"${CMD_PID}" 2>/dev/null || kill -9 "${CMD_PID}" 2>/dev/null || true
                rm -f "${KILL_LOCK}"
                echo "[KillWatcher] 进程已终止, kill_lock 已清理。"
                exit 0
            fi
        done
    ) &
    _KILL_WATCHER_PID=$!
}

_stop_kill_watcher() {
    # 停止后台哨兵子进程
    # 在 run_cmd 正常/异常结束后调用, 防止哨兵子进程泄漏
    if [ -n "${_KILL_WATCHER_PID}" ]; then
        kill "${_KILL_WATCHER_PID}" 2>/dev/null || true
        wait "${_KILL_WATCHER_PID}" 2>/dev/null || true
        _KILL_WATCHER_PID=""
    fi
}


# ============================================================
# 信号捕获: 确保所有退出路径都清理临时文件
# ============================================================
cleanup() {
    _stop_kill_watcher
    # 若 run_cmd 进程组仍在运行 (如 SIGTERM 触发 cleanup 时), 先终止它
    if [ -n "${_RUN_CMD_PID}" ]; then
        kill -TERM -"${_RUN_CMD_PID}" 2>/dev/null || kill -TERM "${_RUN_CMD_PID}" 2>/dev/null || true
    fi
    rm -f "/home/penghongen/run_cmd_${SLURM_JOB_ID}.sh"
    rm -f "/home/penghongen/pre_lock_${SLURM_JOB_ID}"
    rm -f "/home/penghongen/after_lock_${SLURM_JOB_ID}"
    rm -f "/home/penghongen/try_lock_${SLURM_JOB_ID}"
    rm -f "${KILL_LOCK}"
    if type pocket_plus_cleanup_local_cache >/dev/null 2>&1; then
        pocket_plus_cleanup_local_cache
    fi
    echo "[Cleanup] Removed lock/cmd/kill files for job ${SLURM_JOB_ID}."
}
trap cleanup EXIT SIGTERM SIGINT





# ============================================================
# 提前生成指令挂载文件 [特性 1]
# ============================================================
# 若 sbatch 脚本中定义了 write_custom_run_cmd_hook 函数，则优先调用它;
# 否则按默认模板生成, 供用户在 try_lock 期间手动编辑。
# ============================================================

RUN_CMD_FILE="/home/penghongen/run_cmd_${SLURM_JOB_ID}.sh"

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


# ============================================================
# pre_lock / after_lock [特性 2]
# ============================================================

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


# ============================================================
# 主循环: 执行 → 成功/失败处理 → 可选重试
# ============================================================
while true; do
    # 清理上一轮可能残留的 kill_lock 文件, 防止哨兵被 _stop_kill_watcher 提前杀死
    # 而未来得及 rm kill_lock, 导致下一轮新哨兵立即误杀新进程
    rm -f "${KILL_LOCK}"

    RUN_STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
    echo "[Run] Local run stamp: ${RUN_STAMP}"
    
    echo "================================================================================="
    echo "[Attempt] Executing dynamic command from ${RUN_CMD_FILE}..."
    cat "${RUN_CMD_FILE}"
    echo "================================================================================="

    # ---- 以独立进程组启动 run_cmd [特性 4 + 特性 5] ----
    # setsid:  创建独立进程组, 使 kill -9 -PID 能杀掉整个进程树
    #          (包括 bash run_cmd.sh 以及它启动的 python 等所有子进程)
    # stdbuf:  强制 stdout(-oL) 和 stderr(-eL) 为行缓冲模式 [特性 5]
    #          确保 Python 输出在 Slurm 非交互环境下实时写入 .out/.err 日志
    setsid stdbuf -oL -eL bash "${RUN_CMD_FILE}" &
    CMD_PID=$!
    _RUN_CMD_PID="${CMD_PID}"

    # 启动 kill_lock 哨兵 [特性 4]: 后台监控 /home/penghongen/kill_lock_JOBID
    _start_kill_watcher "${CMD_PID}"

    # 前台等待 run_cmd 完成 (阻塞直到 bash run_cmd.sh 退出)
    # 注意: 必须用 && ... || ... 模式, 否则 set -e 会在 wait 返回非零退出码
    # (如被 kill -9 时的 137) 时直接终止整个脚本, 导致跳过 try_lock 逻辑
    wait "${CMD_PID}" && CMD_EXIT=0 || CMD_EXIT=$?

    # run_cmd 已结束, 停止哨兵 (防止后台子进程泄漏)
    _stop_kill_watcher
    _RUN_CMD_PID=""

    if [ "${CMD_EXIT}" -eq 0 ]; then
        echo "================================================================================="
        echo "[Success] Execution finished successfully."
        echo "================================================================================="
        
        # ---- 成功分支 [特性 3]: TRY_AFTER_END 挂起机制 ----
        if is_lock_enabled "${TRY_AFTER_END_ENABLED}" && is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
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
            if is_lock_enabled "${TRY_AFTER_END_ENABLED}" && ! is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
                echo "[Info] TRY_AFTER_END_ENABLED=1 但 LOCK_AFTER_ENABLED=0, 无法挂起, 直接退出。"
            fi
            rm -f "${LOCK_AFTER}"
            rm -f "${LOCK_TRY}"
            break
        fi
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[Error] Execution FAILED! (exit code: ${CMD_EXIT}) Check error logs above."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        if ! is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
            echo "[Stop] after_Lock is disabled, so the job will exit immediately after this failure."
            break
        fi

        # ---- 失败分支: 创建 try_lock, 挂起等待用户干预 ----
        # (也是 kill_lock 哨兵触发后的最终落脚点)
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

# ---- 最终 after_lock 守护: 即使循环退出, 仍等待用户确认释放节点 ----
if is_lock_enabled "${LOCK_AFTER_ENABLED}"; then
    while [ -f "${LOCK_AFTER}" ]; do
        sleep 20
    done
fi

# cleanup() 已由 EXIT trap 自动调用, 无需手动 rm
echo "[Exit] Job finished."
