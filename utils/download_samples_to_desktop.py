import os
import json
import subprocess
from pathlib import Path

def download_samples(json_path: str, ssh_user_host: str, num_samples: int = 30):
    """
    读取 JSON 映射文件, 将远程服务器上的对应文件批量下载到本地桌面上指定的四个文件夹。

    输入参数:
        - json_path: str, 标量, 生成的 JSON 映射文件路径(本地的 json 路径)
        - ssh_user_host: str, 标量, SCP 登录凭据字符串。建议值 'user@192.168.0.1'
        - num_samples: int, 标量, 尝试下载的最大样本组数目。建议值 100

    输出:
        - 无返回值, 副作用是调用 SCP 并写入文件至本地
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到 JSON 映射文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        # list[dict[str, str]], 可变长度, 从 JSON 读出的映射结构
        samples = json.load(f)

    # list[dict[str, str]], 形状为[≤num_samples], 选取的下载子集
    subset = samples[:num_samples]
    print(f"JSON 中存在 {len(samples)} 个样本, 准本开始下载其中的 {len(subset)} 个。")

    # str, 标量, 当前用户桌面路径的绝对地址
    desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')

    # dict[str, str], 标量, JSON中字段 对应 桌面保存文件夹 的绝对路径
    target_dirs = {
        "cif_path": os.path.join(desktop_dir, "原始结构"),
        "map_path": os.path.join(desktop_dir, "EMDB密度图"),
        "sim_cif_path": os.path.join(desktop_dir, "受体模拟密度图"),
        "sim_all_path": os.path.join(desktop_dir, "整体模拟密度图")
    }

    # 首先建立四个月标文件夹
    for key, folder_path in target_dirs.items():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

    # int, 标量, 下载完成数量的迭代计数器
    count = 0
    for s in subset:
        count += 1
        print(f"[{count}/{len(subset)}] 正在同步样本: PDB={s.get('pdb_id', 'Unknown')}, EMD={s.get('emdb_id', 'Unknown')} ...")
        
        for key, folder_path in target_dirs.items():
            # str, 标量, 在服务器上的绝对地址
            remote_path = s.get(key, "")
            if not remote_path:
                continue
                
            # list[str], 标量, SCP 调用列表。使用系统内置命令
            cmd = ["scp", f"{ssh_user_host}:{remote_path}", folder_path]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"文件下载失败, 已忽略, 可能无法连接或源路径 {remote_path} 异常。")

    print(f"全部 {min(len(subset), num_samples)} 个样本的任务执行结束!")


if __name__ == "__main__":
    # 需要先将前面第一步生成的 dataset_mapping.json 从服务器拿到本地，再执行此文件
    # SSH请确保配好了密钥免密登录，或者能够正常手动输入密码
    download_samples(
        json_path=r"C:\Users\15919\Desktop\dataset_mapping.json", 
        ssh_user_host="penghongen@10.102.33.220", 
        num_samples=30
    )
