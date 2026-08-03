#!/usr/bin/env bash

# 本文件由 Job 启动时的 task.sbatch source。它保留四锁和动态 run_cmd。
# 完整模式在每一次实际执行前从 TASK_ROOT 创建或复用 release：
#
# allocations/
# ├── pre_lock_<job-id>       # 仅 --pre_hold 时创建；删除后开始
# ├── try_lock_<job-id>       # 仅 --after_hold 时在执行结束后创建；删除后再次执行
# └── <job-id>/
#     ├── after_lock_<job-id> # 删除后退出并释放 allocation
#     ├── kill_lock_<job-id>  # 人工创建；哨兵终止当前进程组
#     ├── run_cmd_<job-id>.sh # 可在 try_lock 期间改成任意命令
#     ├── out
#     └── err
#
# out/err 的重定向由 task.sbatch 完成；本文件不增加日志轮转或状态文件。

run_allocation() {
    local job_id="${SLURM_JOB_ID:?缺少 SLURM_JOB_ID}"
    local simple_mode="${TASK_SIMPLE_MODE:-0}"
    local pre_hold_mode="${TASK_PRE_HOLD_MODE:-0}"
    local after_hold_mode="${TASK_AFTER_HOLD_MODE:-0}"
    local allocation_root
    local job_directory
    if [[ "${simple_mode}" == "1" ]]; then
        allocation_root="${HOME}/SIMPLE_RUN"
        job_directory="${allocation_root}"
    else
        allocation_root="${TASK_FEEDBACK_ROOT}/allocations"
        job_directory="${allocation_root}/${job_id}"
    fi
    local pre_lock="${allocation_root}/pre_lock_${job_id}"
    local try_lock="${allocation_root}/try_lock_${job_id}"
    local after_lock="${job_directory}/after_lock_${job_id}"
    local kill_lock="${job_directory}/kill_lock_${job_id}"
    local run_cmd="${job_directory}/run_cmd_${job_id}.sh"
    local release_helper="${TASK_ROOT}/训练与运行/runtime/create_release.sh"
    local releases_root="${TASK_FEEDBACK_ROOT}/releases"
    local task_name
    local attempt=0
    local watcher_pid=""
    local command_pid=""
    local last_command_exit=0
    # 正式轮询间隔保持稳定；测试可缩短间隔，但不会改变锁文件语义。
    local lock_poll_seconds="${TASK_LOCK_POLL_SECONDS:-20}"
    local kill_poll_seconds="${TASK_KILL_POLL_SECONDS:-10}"

    task_name="$(basename "${TASK_PATH}" .sh)"
    task_name="${task_name//[^A-Za-z0-9_.-]/_}"

    stop_kill_watcher() {
        if [[ -n "${watcher_pid}" ]]; then
            kill "${watcher_pid}" 2>/dev/null || true
            wait "${watcher_pid}" 2>/dev/null || true
            watcher_pid=""
        fi
    }

    cleanup_allocation() {
        stop_kill_watcher
        if [[ -n "${command_pid}" ]]; then
            kill -TERM -"${command_pid}" 2>/dev/null \
                || kill -TERM "${command_pid}" 2>/dev/null \
                || true
        fi
        rm -f -- "${pre_lock}" "${try_lock}" "${after_lock}" "${kill_lock}" "${run_cmd}"
        printf '[allocation] Job %s 已退出并清理活动锁与动态命令。\n' "${job_id}"
    }
    trap cleanup_allocation EXIT
    trap 'exit 143' SIGTERM
    trap 'exit 130' SIGINT

    start_kill_watcher() {
        local monitored_pid="$1"
        (
            while true; do
                sleep "${kill_poll_seconds}"
                if [[ -f "${kill_lock}" ]]; then
                    printf '[kill_lock] 检测到 %s，终止进程组 %s。\n' \
                        "${kill_lock}" "${monitored_pid}"
                    kill -9 -"${monitored_pid}" 2>/dev/null \
                        || kill -9 "${monitored_pid}" 2>/dev/null \
                        || true
                    rm -f -- "${kill_lock}"
                    exit 0
                fi
            done
        ) &
        watcher_pid=$!
    }

    write_initial_run_cmd() {
        {
            printf '#!/usr/bin/env bash\n'
            printf 'set -euo pipefail\n\n'
            printf '# 本文件可以在 pre_lock 或 try_lock 存在、任务未运行时编辑。\n'
            if [[ "${simple_mode}" == "1" ]]; then
                printf '# simple 模式直接使用当前项目中的任务脚本，不创建 release。\n'
                printf 'exec bash "${TASK_ROOT}"/%q' "${TASK_PATH}"
            else
                printf '# TASK_PROJECT_ROOT 会在每次执行前指向该次刚创建的 release。\n'
                printf 'exec bash "${TASK_PROJECT_ROOT}"/%q' "${TASK_PATH}"
            fi
            if ((${#task_arguments[@]} > 0)); then
                local argument
                for argument in "${task_arguments[@]}"; do
                    printf ' %q' "${argument}"
                done
            fi
            printf '\n'
        } >"${run_cmd}"
        chmod +x "${run_cmd}"
    }

    wait_for_next_action() {
        touch "${try_lock}"
        printf '[allocation] 已创建 %s。\n' "${try_lock}"
        printf '[allocation] --after_hold 正在保留资源；删除 try_lock 再次执行，删除 %s 结束 Job。\n' "${after_lock}"
        while [[ -f "${try_lock}" && -f "${after_lock}" ]]; do
            sleep "${lock_poll_seconds}"
        done
        [[ -f "${after_lock}" ]]
    }

    wait_after_attempt_if_requested() {
        if [[ "${after_hold_mode}" != "1" ]]; then
            printf '[allocation] 未启用 --after_hold；本次执行结束后立即释放 allocation。\n'
            return 1
        fi
        wait_for_next_action
    }

    if [[ "${simple_mode}" == "1" ]]; then
        mkdir -p "${allocation_root}"
    else
        mkdir -p "${job_directory}" "${releases_root}" \
            "${TASK_FEEDBACK_ROOT}/launches/${job_id}"
    fi
    write_initial_run_cmd
    touch "${after_lock}"

    if [[ "${pre_hold_mode}" == "1" ]]; then
        touch "${pre_lock}"
        printf '[allocation] --pre_hold 已创建 %s；删除它后执行初始命令。\n' "${pre_lock}"
        while [[ -f "${pre_lock}" && -f "${after_lock}" ]]; do
            sleep "${lock_poll_seconds}"
        done
        if [[ ! -f "${after_lock}" ]]; then
            return 0
        fi
    fi

    while [[ -f "${after_lock}" ]]; do
        attempt=$((attempt + 1))
        rm -f -- "${kill_lock}"

        local release_project_root=""
        local task_path=""
        local launch_helper=""
        local started_at=""
        local launch_id=""
        local launch_directory=""
        local command_exit=0

        if [[ "${simple_mode}" == "1" ]]; then
            export TASK_PROJECT_ROOT="${TASK_ROOT}"
        else
            # 关键时序：release 在本次 run_cmd 即将执行时才产生。若排队期间或上一次
            # try_lock 期间更新了发布源，本次哈希会得到相应的新 release。
            if [[ ! -f "${release_helper}" ]]; then
                last_command_exit=2
                printf '[allocation][错误] 发布源缺少 release 工具：%s\n' \
                    "${release_helper}" >&2
                if ! wait_after_attempt_if_requested; then
                    break
                fi
                continue
            fi
            if release_project_root="$(
                bash "${release_helper}" "${TASK_ROOT}" "${releases_root}"
            )"; then
                :
            else
                command_exit=$?
                last_command_exit="${command_exit}"
                printf '[allocation][错误] 第 %s 次执行无法创建 release，退出码 %s。\n' \
                    "${attempt}" "${command_exit}" >&2
                if ! wait_after_attempt_if_requested; then
                    break
                fi
                continue
            fi

            export TASK_PROJECT_ROOT="${release_project_root}"
            task_path="${TASK_PROJECT_ROOT}/${TASK_PATH}"
            launch_helper="${TASK_PROJECT_ROOT}/训练与运行/runtime/create_launch.sh"
            if [[ ! -f "${task_path}" || ! -f "${launch_helper}" ]]; then
                last_command_exit=2
                printf '[allocation][错误] release 缺少任务或 launch 工具：%s\n' \
                    "${TASK_PROJECT_ROOT}" >&2
                if ! wait_after_attempt_if_requested; then
                    break
                fi
                continue
            fi
        fi

        export TASK_GPUS="${TASK_GPUS_PER_NODE}"
        export TASK_NNODES="${TASK_NODE_COUNT}"

        if [[ "${simple_mode}" == "1" ]]; then
            :
        else
            started_at="$(date '+%Y%m%dT%H%M%S')"
            launch_id="${task_name}_job${job_id}_${started_at}_a${attempt}"
            export TASK_RUN_STAMP="${launch_id}"
            export EXPERIMENT_FEEDBACK_ROOT="${TASK_FEEDBACK_ROOT}"
            if launch_directory="$(
                bash "${launch_helper}" "${TASK_FEEDBACK_ROOT}" "${launch_id}" \
                    "${run_cmd}" "${TASK_PATH}"
            )"; then
                :
            else
                command_exit=$?
                last_command_exit="${command_exit}"
                printf '[allocation][错误] 第 %s 次执行无法建立 launch，退出码 %s。\n' \
                    "${attempt}" "${command_exit}" >&2
                if ! wait_after_attempt_if_requested; then
                    break
                fi
                continue
            fi
            export TASK_LAUNCH_DIR="${launch_directory}"
        fi

        if [[ "${simple_mode}" == "1" ]]; then
            printf '[allocation] 第 %s 次执行；simple 模式，不创建 release 或 launch。\n' \
                "${attempt}"
        else
            printf '[allocation] 第 %s 次执行；release=%s\n' \
                "${attempt}" "${TASK_PROJECT_ROOT}"
            printf '[allocation] launch=%s\n' "${launch_directory}"
        fi
        printf '[allocation] 动态命令：%s\n' "${run_cmd}"
        cat "${run_cmd}"

        (
            cd "${TASK_PROJECT_ROOT}"
            exec setsid stdbuf -oL -eL bash "${run_cmd}"
        ) &
        command_pid=$!
        start_kill_watcher "${command_pid}"
        wait "${command_pid}" && command_exit=0 || command_exit=$?
        last_command_exit="${command_exit}"
        stop_kill_watcher
        command_pid=""

        if [[ "${command_exit}" -eq 0 ]]; then
            printf '[allocation] 第 %s 次执行成功。\n' "${attempt}"
        else
            printf '[allocation] 第 %s 次执行失败，退出码为 %s。\n' \
                "${attempt}" "${command_exit}" >&2
        fi

        if ! wait_after_attempt_if_requested; then
            break
        fi
    done

    # 正常返回前在局部变量仍有效时清理，再解除 shell 级 trap；否则调用者稍后
    # 退出时，trap 会引用已经离开作用域的局部变量。
    trap - EXIT SIGTERM SIGINT
    cleanup_allocation
    return "${last_command_exit}"
}
