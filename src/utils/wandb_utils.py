import os
# WandB 网络预检与自动回退逻辑
# ==============================================================================
def _check_wandb_connectivity(timeout: float = 10.0) -> bool:
    """
    检查是否能够连接到 WandB 服务器。
    Args:
        - timeout: float, 连接超时时间 (秒)
    Returns:
        - bool, True 表示可以连接, False 表示无法连接
    """
    import socket
    old_timeout = socket.getdefaulttimeout()
    test_sock = None
    try:
        # 尝试连接 WandB API 服务器
        socket.setdefaulttimeout(timeout)
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.connect(("api.wandb.ai", 443))
        return True
    except (socket.timeout, socket.error, OSError):
        return False
    finally:
        if test_sock is not None:
            try:
                test_sock.close()
            except Exception:
                pass
        socket.setdefaulttimeout(old_timeout)

def _setup_wandb_mode(prefer_online: bool = True, verbose: bool = True) -> str:
    """
    根据网络状态和用户配置设置 WandB 模式。
    Args:
        - prefer_online: bool, 用户是否希望使用在线模式 (来自配置 offline=False)
        - verbose: bool, 是否打印日志信息 (DDP 模式下仅 rank 0 应设为 True)
    Returns:
        - str, 'online', 'offline'
    """
    # 如果用户明确指定离线模式，直接使用
    if not prefer_online:
        os.environ["WANDB_MODE"] = "offline"
        return "offline"
    
    # 检查网络连通性
    if verbose:
        print("[Train] 检测 WandB 网络连通性 (Checking WandB connectivity)...")
    if _check_wandb_connectivity(timeout=5.0):
        # 检查 WandB API Key 是否已配置
        try:
            import wandb
            api_key = wandb.api.api_key or os.environ.get("WANDB_API_KEY")
        except Exception:
            api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            if verbose:
                print("[Train] [X] 网络可达但未配置 WandB API Key, 自动切换到离线模式 (No API key, falling back to offline)")
                print("[Train]   请使用 'wandb login' 配置 API Key 以启用在线模式")
            os.environ["WANDB_MODE"] = "offline"
            return "offline"
        if verbose:
            print("[Train] [OK] 网络可达, 使用在线模式 (Network reachable, using online mode)")
        # 在线模式成功时，设置 WANDB_DIR 为临时目录以避免保存本地日志
        import tempfile
        os.environ["WANDB_DIR"] = tempfile.gettempdir()
        if verbose:
            print("[Train]   本地日志已重定向到临时目录 (Local logs redirected to temp)")
        return "online"
    else:
        if verbose:
            print("[Train] [X] 网络不可达, 自动切换到离线模式 (Network unreachable, falling back to offline mode)")
        os.environ["WANDB_MODE"] = "offline"
        return "offline"