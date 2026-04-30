"""
以下是本文件的处理逻辑:
见 readme.md
"""

import json
import os
import random
import pandas as pd
from pathlib import Path

# ============================================================================
# 配置 / Configuration
# ============================================================================

random.seed(42)

# str, EMDB-PDB 映射 csv 路径
EMDB_TO_PDB_CSV_PATH = "/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv"
EMDB_COLUMN_NAME = 'emdb_id'
PDB_COLUMN_NAME = 'fitted_pdbs'
# PDB 的总过滤csv: 直接读取这个.csv里面的所有字母id, 如果之后某个pdb的条目不在其中, 那就先过滤掉(大小写进行统一后再匹配)
PDB_FILTER_CSV = "/home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv"


# emdb\pdb\sim的存放文件夹
EMDB_FOLDER = "/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc"
add_prefix_of_emdb='emd_'   # EMDB具体文件的前缀
PDB_FOLDER = "/storage/chenzhaoyang/cryo_em/CIF_3.5_cc_qscore"
upper_lower_of_pdb='upper'  # PDB具体文件的大小写
SIM_FOLDER = '/storage/penghongen/simulated_receptor_map/'
add_prefix_of_sim='emd_'   # 模拟密度图具体文件的前缀
# PARSED_PDB_ROOT = "/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/"  # 经过PDB_processor处理后的pdb文件夹
# PARSED_PDB_ROOT = "/home/penghongen/My_Project/Data/DATA_v2_mod4/parsed_pdb/"  # 经过PDB_processor处理后的pdb文件夹
# PARSED_PDB_ROOT = "/home/penghongen/My_Project/Data/DATA_v2_raw5/parsed_pdb/"  # 经过PDB_processor处理后的pdb文件夹
PARSED_PDB_ROOT = "/home/penghongen/My_Project/Data/DATA_v2_mod5/parsed_pdb/"  # 经过PDB_processor处理后的pdb文件夹


# list[str], 测试集候选 CSV 文件路径列表, 每个 CSV 读取第一列(PDB_ID)作为一个子测试集
# 当为空列表时, 不从外部指定测试集 PDB, 仍由 MIN_test_ratio 从总样本中随机补全
TEST_CSV = [
  "/home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_1.csv",    # 纯蛋白:随机
  "/home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_2.csv",    # 纯蛋白:均有生物学意义    
  "/home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_3.csv",    # 纯蛋白:均有生物学意义+无金属  
  "/home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_1.csv",         # 存在核酸: 随机
  "/home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_2.csv",         # 存在核酸: 均有生物学意义
  "/home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_3.csv"          # 存在核酸: 均生物学意义+无金属
]
MAX_test_num = None   # int | None, 每个子测试集的最大数量上限
MIN_test_ratio = None # float | None, 每个子测试集样本数 / 总有效样本数 < 此比例时, 从剩余样本随机补全

# Path, 输出目录
# OUTPUT_DIR = Path("/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/")
# OUTPUT_DIR = Path("/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod4/")
# OUTPUT_DIR = Path("/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw5/")
OUTPUT_DIR = Path("/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod5/")

# 划分描述
validation_set_ratio = 0.05  # float, 大验证集样本数 / 总样本数(含测试集) 比例
# float | list[float] | None, 小训练集占大训练集的比例; None 不生成; 命名如 ratio=0.30 → train_030.json
mini_train_set_ratio = [0.10, 0.25, 0.50]
# float | list[float] | None, 小验证集占大验证集的比例; None 不生成; 命名如 ratio=0.30 → valid_030.json
mini_valid_set_ratio = None

# ============================================================================
# 辅助函数 / Helper Functions
# ============================================================================

def save_json(data, path: Path) -> None:
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {path.name} ({len(data)} 样本)")


def fetch_k_samples(data: list, k: int) -> list:
    """从data(List或Dict)中随机选取 k 个样本"""
    if isinstance(data, dict):
        if k >= len(data):
            return data.copy()
        sampled_items = random.sample(list(data.items()), k)
        return dict(sampled_items)
    if k >= len(data):
        return data.copy()
    return random.sample(data, k)



def load_test_pdb_ids_from_csv(csv_path: str, upper_lower_of_pdb: str) -> set[str]:
    """
    从单个 CSV 文件中读取第一列作为测试集 PDB ID。

    输入参数:
        - csv_path: str, 标量, CSV 文件路径
        - upper_lower_of_pdb: str, 标量, 'upper' 或 'lower', 控制返回 PDB ID 的大小写

    输出:
        - pdb_ids: set[str], 从 CSV 第一列解析出的 PDB ID 集合
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"测试集 CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    col_name = df.columns[0]
    # set[str], 存储格式化后的 PDB ID
    pdb_ids = set()
    for val in df[col_name].dropna():
        # str, 去空格、去后缀
        val_str = str(val).strip().split('.')[0]
        if upper_lower_of_pdb == 'upper':
            pdb_ids.add(val_str.upper())
        else:
            pdb_ids.add(val_str.lower())
    print(f"  [信息] 从 {csv_path} 读取了 {len(pdb_ids)} 个测试集候选 PDB ID")
    return pdb_ids


def ratio_to_suffix(ratio: float) -> str:
    """
    将比例值转换为文件名后缀的纯数字字符串。

    输入参数:
        - ratio: float, 标量, 比例值, 取值范围 (0, 1)

    输出:
        - suffix: str, 标量, 3 位纯数字字符串, 如 '030', '005'
    """
    return f"{int(ratio * 100):03d}"

def dict_to_list_of_single_dicts(data_dict: dict[str, str]) -> list[dict[str, str]]:
    """
    将包含多个键值对的字典Dict[str, str]转换为包含多个单键值对字典的列表List[Dict[str, str]]
    - data_dict: dict[str, str], 形状为 {k1:v1, k2:v2, ...}, 原始存储 EMDB-PDB 对应关系的字典[{k1:v1}, {k2:v2}, ...]
    输出:
    - list[dict[str, str]]: list, 形状为 (N,), 长度为 N 的列表，每个元素是形如 {key: value} 的小字典
    """
    return [{k: v} for k, v in data_dict.items()]


def load_pdb_filter_set(csv_path: str, upper_lower_of_pdb: str) -> set[str]:
    """
    读取 PDB 过滤列表。
    
    输入参数:
        - csv_path: str, 标量, PDB_FILTER_CSV 的路径
        - upper_lower_of_pdb: str, 标量, 决定转换为大写('upper')还是小写('lower')
        
    输出:
        - pdb_set: set[str], 格式化后的 PDB ID 集合，若文件不存在则返回 None
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"  [警告] 找不到 PDB 过滤文件: {csv_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        col_name = df.columns[0]
            
        pdb_set = set()
        for val in df[col_name].dropna():
            val_str = str(val).strip()
            val_str = val_str.split('.')[0] # 兼容可能带有扩展名的情况 (如 9j37.cif)
            if upper_lower_of_pdb == 'upper':
                pdb_set.add(val_str.upper())
            else:
                pdb_set.add(val_str.lower())
        print(f"  [信息] 从 {csv_path} 加载了 {len(pdb_set)} 个 PDB ID 用于过滤")
        return pdb_set
    except Exception as e:
        print(f"  [错误] 读取 {csv_path} 失败: {e}")
        return None


def make_EMDB_PDB_dict(emdb_to_pdb_csv_path=EMDB_TO_PDB_CSV_PATH, emdb_column_name=EMDB_COLUMN_NAME, pdb_column_name=PDB_COLUMN_NAME, 
                        add_prefix_of_emdb: str='emd_', upper_lower_of_pdb: str='upper', 
                        exclude_pdb_list: list[str] = None):
    """ 返回 EMDB-PDB 字典. 格式为 Dict[str, str] {'EMD-611081':'9j37', 'EMD-...':'...'}, 在返回时只选第一个pdb(即使对应多个)
    - prefix_of_emdb: 注意, 总是同时支持关于EMDB_ID的好几种格式(按'-'或'_'划分取最后). 但返回的字典中的key = prefix_of_emdb + emdb_id(数字部分)
    - upper_lower_of_pdb: 'upper' 或 'lower', 控制返回字典中 PDB ID 的大小写
    - exclude_pdb_list: list[str], 需要排除的pdb的文件名, 如9j37. 如果一个EMDB的对应PDB与exclude_pdb_list的交集非空, 则不返回该EMDB对应的键值对.注意这是在应用 upper_lower_of_pdb调整了emdb_to_pdb_csv_path之后再检测的.
    """
    df_dict = pd.read_csv(emdb_to_pdb_csv_path)

    # 检查第一行数据的格式来确定emdb_id的分隔符
    if len(df_dict) == 0:
        raise ValueError("CSV文件为空")
    sample_id = df_dict[emdb_column_name].iloc[0]
    if exclude_pdb_list is None:
        exclude_pdb_set = set()
    else:
        exclude_pdb_set = set(exclude_pdb_list)

    if '-' in str(sample_id):   # 如'...-61081'
        df_dict['temp_emdb_id'] = df_dict[emdb_column_name].apply(   # 对每一列(如emd-61081或EMDB-61081)使用
            lambda x:   add_prefix_of_emdb + x.split('-')[-1]
        )
    elif '_' in str(sample_id):   # 如'..._61081'
        df_dict['temp_emdb_id'] = df_dict[emdb_column_name].apply(
            lambda x:   add_prefix_of_emdb + x.split('_')[-1]
        )
    else:
        raise ValueError(f"EMDB_ID 格式无法识别: {sample_id}")


    if upper_lower_of_pdb == "upper":
        df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(
            lambda x: str(x).split(',')[0].strip().upper()    if ',' in str(x)
            else str(x).upper()
        )
    elif upper_lower_of_pdb == 'lower':
        df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(
            lambda x: str(x).split(',')[0].strip().lower()    if ',' in str(x)
            else str(x).lower()
        )
    df_dict['temp_pdbs_list'] = df_dict['temp_pdbs_list'].apply(lambda x: x if x not in exclude_pdb_set else None)


    processed_dict = dict(zip(df_dict['temp_emdb_id'], df_dict['temp_pdbs_list']))
    filtered_dict = {}
    for emdb_id, pdb in processed_dict.items():
        if pdb is not None:
            filtered_dict[emdb_id] = pdb

    return filtered_dict

def get_existing_emdb_pdb(emdb_to_pdb_dict, 
                          emdb_folder=None, 
                          pdb_folder=None, 
                          parsed_pdb_root=None,
                          sim_folder=None,
                          pdb_filter_set=None):
    """ 从emdb_to_pdb_dict中筛选条目, 使得所有筛选后的条目在各个文件夹中都存在.(对于emdb只统一检查数字部分) 
    - emdb_to_pdb_dict: dict[str, str], 返回的 EMDB-PDB 字典
    - emdb_folder: str, emdb文件夹路径
    - pdb_folder: str, pdb文件夹路径
    - parsed_pdb_root: str, parsed_pdb文件夹路径, 由 Pocket_classic\Make_Data\PDB_processor 返回
    - sim_folder: str, 模拟密度图文件夹路径
    - pdb_filter_set: set[str], 从 PDB_FILTER_CSV 读取的过滤集合
    """
    emdb_files_set = {os.path.splitext(f)[0].split('_')[-1].split('-')[-1] for f in os.listdir(emdb_folder)}  # 无前后缀, 如'61081'
    pdb_files_set = {os.path.splitext(f)[0] for f in os.listdir(pdb_folder)}  # 无后缀, 如'9j37'
    parsed_pdb_set = {f.upper() for f in os.listdir(parsed_pdb_root)}  # 无后缀, 如'9j37' #NOTE:这里加了upper
    if sim_folder is not None:
        sim_files_set = {os.path.splitext(f)[0].split('_')[-1].split('-')[-1] for f in os.listdir(sim_folder)}
    else:
        sim_files_set = None

    filtered = {}
    for emdb_id, pdb in emdb_to_pdb_dict.items():

        if pdb_filter_set is not None and pdb not in pdb_filter_set:
            continue
        if pdb not in pdb_files_set:
            continue
        if pdb not in parsed_pdb_set:
            continue

        emdb_id_num = emdb_id.split('_')[-1].split('-')[-1]  # 检查emdb_id是否存在
        if emdb_id_num not in emdb_files_set:
            continue
            
        if sim_files_set is not None and emdb_id_num not in sim_files_set:
            continue

        filtered[emdb_id] = pdb

    return filtered
    
# ============================================================================
# 主函数 / Main Function
# ============================================================================

def main():
    print("开始划分")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_emdb_pdb_dict = make_EMDB_PDB_dict(emdb_to_pdb_csv_path=EMDB_TO_PDB_CSV_PATH, 
                                            emdb_column_name=EMDB_COLUMN_NAME,            
                                            pdb_column_name=PDB_COLUMN_NAME, 
                                            add_prefix_of_emdb=add_prefix_of_emdb,   # NOTE:注意这里的前缀和大小写
                                            upper_lower_of_pdb=upper_lower_of_pdb, 
                                            exclude_pdb_list = None)  # 可能有些emdb_id对应的pdb不存在, 后面会筛选
                                            
    # 读取 PDB_FILTER_CSV
    pdb_filter_set = load_pdb_filter_set(PDB_FILTER_CSV, upper_lower_of_pdb)
    
    all_emdb_pdb_dict = get_existing_emdb_pdb(all_emdb_pdb_dict, 
                                              emdb_folder=EMDB_FOLDER, 
                                              pdb_folder=PDB_FOLDER, 
                                              parsed_pdb_root=PARSED_PDB_ROOT,
                                              sim_folder=SIM_FOLDER,
                                              pdb_filter_set=pdb_filter_set)  # 无空缺版

    # ========== Step 1: 创建全部的.json ==========
    print("\n[Step 1] 创建全部的.json...")
    all_emdb_pdb_json_path = OUTPUT_DIR / "all.json"    # 无空缺版;所有
    save_json(dict_to_list_of_single_dicts(all_emdb_pdb_dict), all_emdb_pdb_json_path)


    # ========== Step 2: 创建测试集.json (多子测试集 + 并集) ==========
    print("\n[Step 2] 创建测试集.json...")
    # list[str | None], 当 TEST_CSV 为空时仍产生一个空候选的子测试集
    csv_list = TEST_CSV if len(TEST_CSV) > 0 else [None]
    # list[dict[str, str]], 收集所有子测试集字典, 最终取并集
    all_sub_test_dicts = []

    for i, csv_path in enumerate(csv_list):
        if csv_path is not None:
            # set[str], 当前 CSV 的候选 PDB ID
            candidate_pdb_ids = load_test_pdb_ids_from_csv(csv_path, upper_lower_of_pdb)
        else:
            candidate_pdb_ids = set()
            print("  [信息] TEST_CSV 为空, 当前子测试集将完全由 MIN_test_ratio 随机补全")

        # dict[str, str], 与总可用样本取交集
        sub_test_dict = {k: v for k, v in all_emdb_pdb_dict.items() if v in candidate_pdb_ids}

        # MAX_test_num 截断
        if MAX_test_num is not None:
            sub_test_dict = fetch_k_samples(data=sub_test_dict, k=MAX_test_num)

        # MIN_test_ratio 补全(针对单个子测试集)
        if MIN_test_ratio is not None:
            # int, 目标测试集数量
            target_test_num = int(len(all_emdb_pdb_dict) * MIN_test_ratio)
            if len(sub_test_dict) < target_test_num:
                # dict[str, str], 排除当前子测试集后的剩余样本
                remaining_dict = {k: v for k, v in all_emdb_pdb_dict.items() if k not in sub_test_dict}
                # int, 需要额外补充的数量
                needed = target_test_num - len(sub_test_dict)
                extra_test_dict = fetch_k_samples(data=remaining_dict, k=needed)
                sub_test_dict.update(extra_test_dict)

        save_json(dict_to_list_of_single_dicts(sub_test_dict), OUTPUT_DIR / f"test_{i}.json")
        all_sub_test_dicts.append(sub_test_dict)

    # dict[str, str], 所有子测试集取并集 → test.json
    test_emdb_pdb_dict = {}
    for d in all_sub_test_dicts:
        test_emdb_pdb_dict.update(d)
    save_json(dict_to_list_of_single_dicts(test_emdb_pdb_dict), OUTPUT_DIR / "test.json")

    # dict[str, str], 无空缺+训练验证集(排除测试集并集)
    train_val_emdb_pdb_dict = {k: v for k, v in all_emdb_pdb_dict.items() if k not in test_emdb_pdb_dict}


    # ========== Step 3: 创建大验证集.json ==========
    print("\n[Step 3] 创建大验证集.json...")             # 无空缺+验证集
    num_validation_set = int(len(all_emdb_pdb_dict) * validation_set_ratio)
    val_emdb_pdb_dict = fetch_k_samples(train_val_emdb_pdb_dict, k=num_validation_set)
    val_emdb_pdb_json_path = OUTPUT_DIR / "val.json"
    save_json(dict_to_list_of_single_dicts(val_emdb_pdb_dict), val_emdb_pdb_json_path)
    

    # ========== Step 4: 大创建训练集.json ==========
    # 在train_val_emdb_pdb_dict中且不在val_emdb_pdb_dict中的样本
    print("\n[Step 4] 创建训练集.json...")               # 无空缺+训练集
    train_emdb_pdb_dict = {k: v for k, v in train_val_emdb_pdb_dict.items() if k not in val_emdb_pdb_dict}  
    train_emdb_pdb_json_path = OUTPUT_DIR / "train.json"
    save_json(dict_to_list_of_single_dicts(train_emdb_pdb_dict), train_emdb_pdb_json_path)


    # ========== Step 5: 创建小验证集.json ==========
    # list[float], 统一为列表处理
    if mini_valid_set_ratio is None:
        valid_ratios = []
    elif isinstance(mini_valid_set_ratio, (int, float)):
        valid_ratios = [float(mini_valid_set_ratio)]
    else:
        valid_ratios = list(mini_valid_set_ratio)

    if valid_ratios:
        print(f"\n[Step 5] 创建小验证集.json... (ratios={valid_ratios})")
        for ratio in valid_ratios:
            # int, 小验证集样本数
            num_mini_val = int(len(val_emdb_pdb_dict) * ratio)
            mini_val_dict = fetch_k_samples(val_emdb_pdb_dict, k=num_mini_val)
            # str, 文件名后缀, 如 '030'
            suffix = ratio_to_suffix(ratio)
            save_json(dict_to_list_of_single_dicts(mini_val_dict), OUTPUT_DIR / f"valid_{suffix}.json")
    else:
        print("\n[Step 5] 跳过小验证集(mini_valid_set_ratio=None)")


    # ========== Step 6: 创建小训练集.json ==========
    # list[float], 统一为列表处理
    if mini_train_set_ratio is None:
        train_ratios = []
    elif isinstance(mini_train_set_ratio, (int, float)):
        train_ratios = [float(mini_train_set_ratio)]
    else:
        train_ratios = list(mini_train_set_ratio)

    if train_ratios:
        print(f"\n[Step 6] 创建小训练集.json... (ratios={train_ratios})")
        for ratio in train_ratios:
            # int, 小训练集样本数
            num_mini_train = int(len(train_emdb_pdb_dict) * ratio)
            mini_train_dict = fetch_k_samples(train_emdb_pdb_dict, k=num_mini_train)
            # str, 文件名后缀, 如 '030'
            suffix = ratio_to_suffix(ratio)
            save_json(dict_to_list_of_single_dicts(mini_train_dict), OUTPUT_DIR / f"train_{suffix}.json")
    else:
        print("\n[Step 6] 跳过小训练集(mini_train_set_ratio=None)")
    


    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print(f"完成! 请检查目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 打印汇总
    print("\n划分汇总:")
    print(f"  all.json: {len(all_emdb_pdb_dict)} 样本")
    print(f"  train.json: {len(train_emdb_pdb_dict)} 样本")
    print(f"  val.json: {len(val_emdb_pdb_dict)} 样本")
    print(f"  test.json (并集): {len(test_emdb_pdb_dict)} 样本")
    for i, d in enumerate(all_sub_test_dicts):
        print(f"    test_{i}.json: {len(d)} 样本")
    for ratio in train_ratios:
        suffix = ratio_to_suffix(ratio)
        print(f"  train_{suffix}.json: {int(len(train_emdb_pdb_dict) * ratio)} 样本")
    for ratio in valid_ratios:
        suffix = ratio_to_suffix(ratio)
        print(f"  valid_{suffix}.json: {int(len(val_emdb_pdb_dict) * ratio)} 样本")


if __name__ == "__main__":
    main()


# 划分 v0 数据集的时候: 
"""
conda activate vnegnn
/home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/Make_Data/split_data/generate_full_json.py
(base) [penghongen@master ~]$ conda activate vnegnn
WARNING: overwriting environment variables set in the machine
overwriting variable {'LD_LIBRARY_PATH'}
(vnegnn) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/Make_Data/split_data/generate_full_json.py
开始划分

[Step 1] 创建全部的.json...
  已保存: all.json (2680 样本)

[Step 2] 创建测试集.json...
  已保存: test.json (268 样本)

[Step 3] 创建大验证集.json...
  已保存: val.json (268 样本)

[Step 4] 创建训练集.json...
  已保存: train.json (2144 样本)

[Step 5] 创建小验证集.json... (n=10)
  已保存: val_0.json (27 样本)
  已保存: val_1.json (27 样本)
  已保存: val_2.json (27 样本)
  已保存: val_3.json (27 样本)
  已保存: val_4.json (27 样本)
  已保存: val_5.json (27 样本)
  已保存: val_6.json (27 样本)
  已保存: val_7.json (27 样本)
  已保存: val_8.json (26 样本)
  已保存: val_9.json (26 样本)

[Step 6] 创建小训练集.json... (n=10)
  已保存: train_0.json (215 样本)
  已保存: train_1.json (215 样本)
  已保存: train_2.json (215 样本)
  已保存: train_3.json (215 样本)
  已保存: train_4.json (214 样本)
  已保存: train_5.json (214 样本)
  已保存: train_6.json (214 样本)
  已保存: train_7.json (214 样本)
  已保存: train_8.json (214 样本)
  已保存: train_9.json (214 样本)

============================================================
完成! 请检查目录: /home/penghongen/My_Project/Data/split/3.5_cc_qscore_v0
============================================================

划分汇总:
  train.json: 2144 样本
  val.json: 268 样本
  test.json: 268 样本
(vnegnn) [penghongen@master ~]$ 
"""


# 划分 v1 数据集的时候: 
"""
(vnegnn) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/Make_Data/split_data/generate_full_json.py
开始划分

[Step 1] 创建全部的.json...
  已保存: all.json (2680 样本)

[Step 2] 创建测试集.json...
  已保存: test.json (268 样本)

[Step 3] 创建大验证集.json...
  已保存: val.json (268 样本)

[Step 4] 创建训练集.json...
  已保存: train.json (2144 样本)

[Step 5] 创建小验证集.json... (n=10)
  已保存: val_0.json (27 样本)
  已保存: val_1.json (27 样本)
  已保存: val_2.json (27 样本)
  已保存: val_3.json (27 样本)
  已保存: val_4.json (27 样本)
  已保存: val_5.json (27 样本)
  已保存: val_6.json (27 样本)
  已保存: val_7.json (27 样本)
  已保存: val_8.json (26 样本)
  已保存: val_9.json (26 样本)

[Step 6] 创建小训练集.json... (n=10)
  已保存: train_0.json (215 样本)
  已保存: train_1.json (215 样本)
  已保存: train_2.json (215 样本)
  已保存: train_3.json (215 样本)
  已保存: train_4.json (214 样本)
  已保存: train_5.json (214 样本)
  已保存: train_6.json (214 样本)
  已保存: train_7.json (214 样本)
  已保存: train_8.json (214 样本)
  已保存: train_9.json (214 样本)

============================================================
完成! 请检查目录: /home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1
============================================================

划分汇总:
  train.json: 2144 样本
  val.json: 268 样本
  test.json: 268 样本
(vnegnn) [penghongen@master ~]$ 
"""







# v2_raw4
"""
conda activate Pocket_Plus_centos7_cu121_allgpu
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
(base) [penghongen@master ~]$ conda activate Pocket_Plus_centos7_cu121_allgpu
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
开始划分
  [信息] 从 /home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv 加载了 6544 个 PDB ID 用于过滤

[Step 1] 创建全部的.json...
  已保存: all.json (5390 样本)

[Step 2] 创建测试集.json...
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_1.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_0.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_2.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_1.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_3.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_2.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_1.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_3.json (54 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_2.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_4.json (54 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_3.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_5.json (54 样本)
  已保存: test.json (969 样本)

[Step 3] 创建大验证集.json...
  已保存: val.json (269 样本)

[Step 4] 创建训练集.json...
  已保存: train.json (4152 样本)

[Step 5] 跳过小验证集(mini_valid_set_ratio=None)

[Step 6] 创建小训练集.json... (ratios=[0.1, 0.25, 0.5])
  已保存: train_010.json (415 样本)
  已保存: train_025.json (1038 样本)
  已保存: train_050.json (2076 样本)

============================================================
完成! 请检查目录: /home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4
============================================================

划分汇总:
  all.json: 5390 样本
  train.json: 4152 样本
  val.json: 269 样本
  test.json (并集): 969 样本
    test_0.json: 269 样本
    test_1.json: 269 样本
    test_2.json: 269 样本
    test_3.json: 54 样本
    test_4.json: 54 样本
    test_5.json: 54 样本
  train_010.json: 415 样本
  train_025.json: 1038 样本
  train_050.json: 2076 样本
"""


# v2_mod4
"""
conda activate Pocket_Plus_centos7_cu121_allgpu
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
(base) [penghongen@master ~]$ conda activate Pocket_Plus_centos7_cu121_allgpu
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
开始划分
  [信息] 从 /home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv 加载了 6544 个 PDB ID 用于过滤

[Step 1] 创建全部的.json...
  已保存: all.json (5285 样本)

[Step 2] 创建测试集.json...
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_1.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_0.json (264 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_2.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_1.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_3.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_2.json (263 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_1.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_3.json (52 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_2.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_4.json (53 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_3.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_5.json (53 样本)
  已保存: test.json (954 样本)

[Step 3] 创建大验证集.json...
  已保存: val.json (264 样本)

[Step 4] 创建训练集.json...
  已保存: train.json (4067 样本)

[Step 5] 跳过小验证集(mini_valid_set_ratio=None)

[Step 6] 创建小训练集.json... (ratios=[0.1, 0.25, 0.5])
  已保存: train_010.json (406 样本)
  已保存: train_025.json (1016 样本)
  已保存: train_050.json (2033 样本)

============================================================
完成! 请检查目录: /home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod4
============================================================

划分汇总:
  all.json: 5285 样本
  train.json: 4067 样本
  val.json: 264 样本
  test.json (并集): 954 样本
    test_0.json: 264 样本
    test_1.json: 269 样本
    test_2.json: 263 样本
    test_3.json: 52 样本
    test_4.json: 53 样本
    test_5.json: 53 样本
  train_010.json: 406 样本
  train_025.json: 1016 样本
  train_050.json: 2033 样本
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ 
"""


# v2_raw5
"""
conda activate Pocket_Plus_centos7_cu121_allgpu
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
(base) [penghongen@master ~]$ conda activate Pocket_Plus_centos7_cu121_allgpu
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
开始划分
  [信息] 从 /home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv 加载了 6544 个 PDB ID 用于过滤

[Step 1] 创建全部的.json...
  已保存: all.json (5395 样本)

[Step 2] 创建测试集.json...
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_1.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_0.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_2.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_1.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_3.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_2.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_1.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_3.json (54 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_2.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_4.json (54 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_3.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_5.json (54 样本)
  已保存: test.json (969 样本)

[Step 3] 创建大验证集.json...
  已保存: val.json (269 样本)

[Step 4] 创建训练集.json...
  已保存: train.json (4157 样本)

[Step 5] 跳过小验证集(mini_valid_set_ratio=None)

[Step 6] 创建小训练集.json... (ratios=[0.1, 0.25, 0.5])
  已保存: train_010.json (415 样本)
  已保存: train_025.json (1039 样本)
  已保存: train_050.json (2078 样本)

============================================================
完成! 请检查目录: /home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw5
============================================================

划分汇总:
  all.json: 5395 样本
  train.json: 4157 样本
  val.json: 269 样本
  test.json (并集): 969 样本
    test_0.json: 269 样本
    test_1.json: 269 样本
    test_2.json: 269 样本
    test_3.json: 54 样本
    test_4.json: 54 样本
    test_5.json: 54 样本
  train_010.json: 415 样本
  train_025.json: 1039 样本
  train_050.json: 2078 样本
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ 
"""


# v2_mod5
"""
conda activate Pocket_Plus_centos7_cu121_allgpu
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
(base) [penghongen@master ~]$ conda activate Pocket_Plus_centos7_cu121_allgpu
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/Make_Data/split_data/generate_full_json.py
开始划分
  [信息] 从 /home/penghongen/My_Project/Data/CIF_3.5_cc_qscore.csv 加载了 6544 个 PDB ID 用于过滤

[Step 1] 创建全部的.json...
  已保存: all.json (5391 样本)

[Step 2] 创建测试集.json...
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_1.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_0.json (268 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_2.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_1.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_protein_3.csv 读取了 269 个测试集候选 PDB ID
  已保存: test_2.json (269 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_1.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_3.json (54 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_2.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_4.json (54 样本)
  [信息] 从 /home/penghongen/My_Project/Data/test_csv/Q_BioLip_na_3.csv 读取了 54 个测试集候选 PDB ID
  已保存: test_5.json (54 样本)
  已保存: test.json (968 样本)

[Step 3] 创建大验证集.json...
  已保存: val.json (269 样本)

[Step 4] 创建训练集.json...
  已保存: train.json (4154 样本)

[Step 5] 跳过小验证集(mini_valid_set_ratio=None)

[Step 6] 创建小训练集.json... (ratios=[0.1, 0.25, 0.5])
  已保存: train_010.json (415 样本)
  已保存: train_025.json (1038 样本)
  已保存: train_050.json (2077 样本)

============================================================
完成! 请检查目录: /home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod5
============================================================

划分汇总:
  all.json: 5391 样本
  train.json: 4154 样本
  val.json: 269 样本
  test.json (并集): 968 样本
    test_0.json: 268 样本
    test_1.json: 269 样本
    test_2.json: 269 样本
    test_3.json: 54 样本
    test_4.json: 54 样本
    test_5.json: 54 样本
  train_010.json: 415 样本
  train_025.json: 1038 样本
  train_050.json: 2077 样本
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ 
"""
