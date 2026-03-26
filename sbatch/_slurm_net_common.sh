#!/bin/bash

# 统一解析训练通信网卡，确保 MASTER_ADDR、NCCL、Gloo 走同一张网卡。
Pocket_Plus_dist_is_excluded_ifname() {
    case "$1" in
        ""|lo|docker*|veth*|virbr*|br-*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}


Pocket_Plus_dist_list_up_ifnames() {
    if ! command -v ip >/dev/null 2>&1; then
        echo "[Dist] 当前节点缺少 ip 命令，无法解析通信网卡" >&2
        return 1
    fi
    ip -o link show up 2>/dev/null | awk -F': ' '{print $2}' | cut -d'@' -f1
}


Pocket_Plus_dist_pick_ifname() {
    local preferred="$1"
    local ifname

    if [ -n "$preferred" ]; then
        if Pocket_Plus_dist_is_excluded_ifname "$preferred"; then
            echo "[Dist] 指定的通信网卡 '$preferred' 不允许使用, 尝试自动回退..." >&2
        elif ! ip link show "$preferred" >/dev/null 2>&1; then
            echo "[Dist] 指定的通信网卡 '$preferred' 不存在, 尝试自动回退..." >&2
        else
            printf '%s\n' "$preferred"
            return 0
        fi
    fi

    if ip link show ib0 >/dev/null 2>&1 && ! Pocket_Plus_dist_is_excluded_ifname "ib0"; then
        printf '%s\n' "ib0"
        return 0
    fi

    while IFS= read -r ifname; do
        case "$ifname" in
            ib*)
                if [ "$ifname" != "ib0" ] && ! Pocket_Plus_dist_is_excluded_ifname "$ifname"; then
                    printf '%s\n' "$ifname"
                    return 0
                fi
                ;;
        esac
    done < <(Pocket_Plus_dist_list_up_ifnames)

    while IFS= read -r ifname; do
        case "$ifname" in
            en*)
                if ! Pocket_Plus_dist_is_excluded_ifname "$ifname"; then
                    printf '%s\n' "$ifname"
                    return 0
                fi
                ;;
        esac
    done < <(Pocket_Plus_dist_list_up_ifnames)

    while IFS= read -r ifname; do
        case "$ifname" in
            eth*)
                if ! Pocket_Plus_dist_is_excluded_ifname "$ifname"; then
                    printf '%s\n' "$ifname"
                    return 0
                fi
                ;;
        esac
    done < <(Pocket_Plus_dist_list_up_ifnames)

    echo "[Dist] 未找到可用的通信网卡" >&2
    return 1
}


Pocket_Plus_dist_get_ipv4_by_ifname() {
    local ifname="$1"
    ip -o -4 addr show dev "$ifname" 2>/dev/null | awk 'NR==1 {print $4}' | cut -d/ -f1
}


Pocket_Plus_dist_refresh_run_stamp() {
    local stamp
    stamp="$(date +%Y-%m-%d_%H-%M-%S)"
    export Pocket_Plus_RUN_STAMP="$stamp"
    export POCKET_RUN_STAMP="$stamp"
}


Pocket_Plus_dist_count_device_list() {
    local raw="$1"
    local item
    local count=0

    raw="${raw// /}"
    case "$raw" in
        ""|"N/A"|"n/a"|"null"|"NULL"|"None"|"none")
            return 1
            ;;
    esac

    IFS=',' read -r -a _pocket_dist_items <<< "$raw"
    for item in "${_pocket_dist_items[@]}"; do
        if [ -n "$item" ]; then
            count=$((count + 1))
        fi
    done

    if [ "$count" -le 0 ]; then
        return 1
    fi

    printf '%s\n' "$count"
}


Pocket_Plus_dist_parse_gpu_count_hint() {
    local raw="$1"
    raw="${raw// /}"

    case "$raw" in
        ""|"N/A"|"n/a"|"null"|"NULL"|"None"|"none")
            return 1
            ;;
    esac

    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$raw"
        return 0
    fi

    if [[ "$raw" =~ ^([0-9]+)\(x[0-9]+\)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    if [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    if [[ "$raw" =~ ^[^:]+:([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    return 1
}


Pocket_Plus_dist_detect_gpus_per_node() {
    local count

    for raw in \
        "${SLURM_STEP_GPUS:-}" \
        "${CUDA_VISIBLE_DEVICES:-}"
    do
        count=$(Pocket_Plus_dist_count_device_list "$raw") || continue
        if [ "$count" -gt 0 ]; then
            printf '%s\n' "$count"
            return 0
        fi
    done

    for raw in \
        "${SLURM_GPUS_ON_NODE:-}" \
        "${SLURM_GPUS_PER_NODE:-}" \
        "${SLURM_STEP_GPUS_PER_NODE:-}"
    do
        count=$(Pocket_Plus_dist_parse_gpu_count_hint "$raw") || continue
        if [ "$count" -gt 0 ]; then
            printf '%s\n' "$count"
            return 0
        fi
    done

    if command -v nvidia-smi >/dev/null 2>&1; then
        count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NF {count += 1} END {print count + 0}')
        if [ -n "$count" ] && [ "$count" -gt 0 ]; then
            printf '%s\n' "$count"
            return 0
        fi
    fi

    return 1
}


Pocket_Plus_dist_init_env() {
    local master_info
    local resolved_ifname
    local master_ip

    if [ -z "$Pocket_Plus_DIST_HELPER_PATH" ]; then
        echo "[Dist] Pocket_Plus_DIST_HELPER_PATH 未设置，无法继续初始化" >&2
        return 1
    fi

    export MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_info=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" --export=ALL bash -lc "
        set -e
        source \"$Pocket_Plus_DIST_HELPER_PATH\"
        resolved_ifname=\$(Pocket_Plus_dist_pick_ifname \"\${SLURM_NET_DIST_IFNAME:-}\")
        master_ip=\$(Pocket_Plus_dist_get_ipv4_by_ifname \"\$resolved_ifname\")
        if [ -z \"\$master_ip\" ]; then
            echo \"[Dist] 网卡 '\$resolved_ifname' 没有可用 IPv4 地址\" >&2
            exit 1
        fi
        printf '%s %s\n' \"\$resolved_ifname\" \"\$master_ip\"
    ") || {
        local status=$?
        echo "[Dist] 无法通过 srun 在 master 节点启动探测步骤 (exit=$status)" >&2
        echo "[Dist] 如果 stderr 中出现 'Job credential expired'，这通常是 Slurm/MUNGE/节点时钟同步问题，不是通信网卡本身初始化失败" >&2
        return "$status"
    }

    read -r resolved_ifname master_ip <<< "$master_info"
    if [ -z "$resolved_ifname" ] || [ -z "$master_ip" ]; then
        echo "[Dist] 无法解析 master 节点通信网卡或 IPv4 地址" >&2
        return 1
    fi

    export SLURM_NET_DIST_IFNAME="$resolved_ifname"
    export MASTER_ADDR="$master_ip"
    export MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))

    if [ -z "$NCCL_SOCKET_IFNAME" ]; then
        export NCCL_SOCKET_IFNAME="$SLURM_NET_DIST_IFNAME"
    fi
    if [ -z "$GLOO_SOCKET_IFNAME" ]; then
        export GLOO_SOCKET_IFNAME="$SLURM_NET_DIST_IFNAME"
    fi
}


Pocket_Plus_dist_preflight() {
    if [ -z "$Pocket_Plus_DIST_HELPER_PATH" ]; then
        echo "[Preflight] Pocket_Plus_DIST_HELPER_PATH 未设置" >&2
        return 1
    fi

    srun --label --kill-on-bad-exit=1 --export=ALL bash -lc "
        set -e
        source \"$Pocket_Plus_DIST_HELPER_PATH\"
        current_ifname=\$(Pocket_Plus_dist_pick_ifname \"\${SLURM_NET_DIST_IFNAME:-}\")
        current_ip=\$(Pocket_Plus_dist_get_ipv4_by_ifname \"\$current_ifname\")
        if [ -z \"\$current_ip\" ]; then
            echo \"[Preflight] 网卡 '\$current_ifname' 没有 IPv4 地址\" >&2
            exit 1
        fi
        route_line=\$(ip route get \"$MASTER_ADDR\" 2>&1 | head -n 1)
        if [ -z \"\$route_line\" ]; then
            echo \"[Preflight] 无法解析到 MASTER_ADDR=$MASTER_ADDR 的路由\" >&2
            exit 1
        fi
        if [ \"\$current_ip\" != \"$MASTER_ADDR\" ]; then
            case \" \$route_line \" in
                *\" dev \$current_ifname \"*)
                    ;;
                *)
                    echo \"[Preflight] 到 MASTER_ADDR=$MASTER_ADDR 的路由未使用网卡 '\$current_ifname': \$route_line\" >&2
                    exit 1
                    ;;
            esac
        fi
        echo \"[Preflight] host=\$(hostname) nodeid=\${SLURM_NODEID:-NA} ifname=\$current_ifname ip=\$current_ip master=$MASTER_ADDR route=\$route_line\"
    " || {
        local status=$?
        echo "[Preflight] srun 预检步骤启动失败 (exit=$status)" >&2
        echo "[Preflight] 如果 stderr 中出现 'Job credential expired'，请重新提交作业，或联系管理员检查 slurmd/munged/NTP/节点时钟同步" >&2
        return "$status"
    }
}
