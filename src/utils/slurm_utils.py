import os
import shutil
import socket
import subprocess


def _normalize_ifname(name: str) -> str:
    return name.split("@", 1)[0].strip()


def _is_excluded_ifname(name: str) -> bool:
    if not name:
        return True
    name = _normalize_ifname(name)
    return (
        name == "lo"
        or name.startswith("docker")
        or name.startswith("veth")
        or name.startswith("virbr")
        or name.startswith("br-")
    )


def _parse_explicit_ifnames(raw: str):
    if not raw:
        return []
    raw = raw.strip()
    if not raw or raw.startswith("^"):
        return []
    return [_normalize_ifname(iface) for iface in raw.split(",") if iface.strip()]


def _parse_excluded_ifnames(*raw_values: str):
    exclude_set = {"lo"}
    for raw in raw_values:
        raw = (raw or "").strip()
        if not raw.startswith("^"):
            continue
        for iface in raw[1:].split(","):
            iface = _normalize_ifname(iface)
            if iface:
                exclude_set.add(iface)
    return exclude_set


def _list_all_ifnames():
    try:
        import netifaces
        all_ifaces = netifaces.interfaces()
    except ImportError:
        try:
            all_ifaces = os.listdir("/sys/class/net")
        except FileNotFoundError:
            all_ifaces = []
    normalized = []
    seen = set()
    for iface in all_ifaces:
        iface = _normalize_ifname(iface)
        if iface and iface not in seen:
            normalized.append(iface)
            seen.add(iface)
    return normalized


def _get_route_ifname(master_addr: str):
    master_addr = (master_addr or "").strip()
    if not master_addr or shutil.which("ip") is None:
        return None
    try:
        result = subprocess.run(
            ["ip", "route", "get", master_addr],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    tokens = result.stdout.strip().split()
    for index, token in enumerate(tokens[:-1]):
        if token == "dev":
            return _normalize_ifname(tokens[index + 1])
    return None


def _build_fallback_ifnames(all_ifaces, exclude_set):
    ordered = []
    seen = set()

    def _append(ifname: str):
        ifname = _normalize_ifname(ifname)
        if (
            not ifname
            or ifname in seen
            or ifname in exclude_set
            or _is_excluded_ifname(ifname)
            or ifname not in all_ifaces
        ):
            return
        ordered.append(ifname)
        seen.add(ifname)

    _append("ib0")
    for prefix in ("ib", "en", "eth"):
        for iface in all_ifaces:
            if prefix == "ib" and iface == "ib0":
                continue
            if iface.startswith(prefix):
                _append(iface)
    for iface in all_ifaces:
        _append(iface)
    return ordered


def _set_gloo_ifname(chosen: str, source: str):
    os.environ["GLOO_SOCKET_IFNAME"] = chosen
    os.environ["SLURM_NET_GLOO_IFNAME_SOURCE"] = source
    rank_hint = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    if rank_hint in ("", "0"):
        print(f"[Train] GLOO_SOCKET_IFNAME 已设置为 '{chosen}' (来源: {source})")


def fix_gloo_socket_ifname():
    """自动修正 GLOO_SOCKET_IFNAME，优先复用与 MASTER_ADDR 一致的通信网卡。"""
    gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME", "")
    nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "")
    slurm_net_ifname = os.environ.get("SLURM_NET_DIST_IFNAME", "")
    master_addr = os.environ.get("MASTER_ADDR", "")
    exclude_set = _parse_excluded_ifnames(gloo_ifname, nccl_ifname)
    all_ifaces = _list_all_ifnames()
    route_ifname = _get_route_ifname(master_addr)
    candidates = []

    for iface in _parse_explicit_ifnames(gloo_ifname):
        candidates.append((iface, "显式 GLOO_SOCKET_IFNAME"))
    for iface in _parse_explicit_ifnames(slurm_net_ifname):
        candidates.append((iface, "显式 SLURM_NET_DIST_IFNAME"))
    for iface in _parse_explicit_ifnames(nccl_ifname):
        candidates.append((iface, "显式 NCCL_SOCKET_IFNAME"))
    if route_ifname:
        candidates.append((route_ifname, f"MASTER_ADDR={master_addr} 的路由结果"))
    for iface in _build_fallback_ifnames(all_ifaces, exclude_set):
        candidates.append((iface, "自动兜底扫描"))

    seen = set()
    for iface, source in candidates:
        iface = _normalize_ifname(iface)
        if not iface or iface in seen:
            continue
        seen.add(iface)
        if iface in exclude_set or _is_excluded_ifname(iface):
            if source != "自动兜底扫描":
                print(f"[Train] 跳过接口 '{iface}'，因为它在排除列表中 (来源: {source})")
            continue
        if iface not in all_ifaces:
            if source != "自动兜底扫描":
                print(f"[Train] 跳过接口 '{iface}'，因为当前节点不存在该接口 (来源: {source})")
            continue
        _set_gloo_ifname(iface, source)
        return

    os.environ.pop("GLOO_SOCKET_IFNAME", None)
    os.environ["SLURM_NET_GLOO_IFNAME_SOURCE"] = "Gloo 自动探测"
    rank_hint = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    if rank_hint in ("", "0"):
        print("[Train] 未找到合适网络接口，已清除 GLOO_SOCKET_IFNAME，由 Gloo 自动探测")


def log_distributed_launch_state(stage: str):
    host = socket.gethostname()
    slurm_procid = os.environ.get("SLURM_PROCID", "NA")
    rank = os.environ.get("RANK", "NA")
    local_rank = os.environ.get("LOCAL_RANK", "NA")
    world_size = os.environ.get("WORLD_SIZE", "NA")
    run_stamp = os.environ.get("POCKET_RUN_STAMP", "NA")
    master_addr = os.environ.get("MASTER_ADDR", "NA")
    master_port = os.environ.get("MASTER_PORT", "NA")
    gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME", "AUTO")
    nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "AUTO")
    gloo_source = os.environ.get("SLURM_NET_GLOO_IFNAME_SOURCE", "未知")
    print(
        f"[Train] {stage}: host={host}, slurm_procid={slurm_procid}, "
        f"rank={rank}, local_rank={local_rank}, world_size={world_size}, run_stamp={run_stamp}, "
        f"master={master_addr}:{master_port}, gloo_ifname={gloo_ifname}, "
        f"nccl_ifname={nccl_ifname}, gloo_source={gloo_source}"
    )
