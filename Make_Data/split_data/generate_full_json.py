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

# emdb\pdb的存放文件夹
EMDB_FOLDER = "/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc"
add_prefix_of_emdb='emd_'   # EMDB具体文件的前缀
PDB_FOLDER = "/storage/chenzhaoyang/cryo_em/PDB_3.5_cc_qscore"
upper_lower_of_pdb='upper'  # PDB具体文件的大小写
PARSED_PDB_ROOT = "/home/penghongen/My_Project/Data/DATA_v1/parsed_pdb/"  # 经过PDB_processor处理后的pdb文件夹

# 预定的测试集文件(将在划分时用到)
TEST_PDB_FOLDER = "/home/penghongen/My_Project/Data/cryatom_output_cif/"
MAX_test_num = 300   # 测试集的最大数量. 如果不为None, 那么会在TEST_PDB_FOLDER中随机选取MAX_test_num个pdb放入test.json, 其它的视为训练/验证集放入 train.json或val.json
MIN_test_ratio = 0.1 # 如果它不为None, 且原本测试集样本数 / 所有有效样本数(all.json条目数 或 all_emdb_pdb_dict长度) < MIN_test_ratio, 那么就先把测试集数目补全到这个比例, 再划分 训练/验证集合

# Path, 输出目录
OUTPUT_DIR = Path("/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/")
return_pdb_mode='Dict[str, str]'

# 划分描述
validation_set_ratio = 0.1  # 大验证集样本数 / 总样本数(含测试集) 比例
num_train_parts = 10        # 小训练集对大训练集的划分份数
num_val_parts = 10          # 小验证集对大验证集的划分份数

# ============================================================================
# 辅助函数 / Helper Functions
# ============================================================================

def save_json(data, path: Path) -> None:
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {path.name} ({len(data)} 样本)")

def split_into_n_parts(data: list, n: int) -> list:
    """
    将数据随机均分为 n 份 (不重不漏; 支持列表/字典)
    
    输入: data (list或dict), n (int)
    输出: List[list或dict], n 个子列表或子字典
    """
    if isinstance(data, dict):
        # shuffled: list[tuple], 形状为 (total, 2), 包含字典所有 (key, value) 元组的列表
        shuffled = list(data.items())
        random.shuffle(shuffled)

        total = len(shuffled)
        base_size = total // n
        remainder = total % n

        parts = []
        start = 0
        for i in range(n):
            size = base_size + (1 if i < remainder else 0)
            parts.append(dict(shuffled[start: start + size]))  # dict()构造字典, 传入List[tuple]
            start += size
        return parts


    shuffled = data.copy()
    random.shuffle(shuffled)
    total = len(shuffled)
    base_size = total // n
    remainder = total % n
    parts = []
    start = 0
    for i in range(n):
        size = base_size + (1 if i < remainder else 0)
        parts.append(shuffled[start: start + size])
        start += size
    return parts

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



def collect_test_pdb_ids(test_pdb_folder: str, upper_lower_of_pdb: str='upper') -> set[str]:
    """
    从测试集文件夹中收集所有的 PDB ID (去掉文件后缀)
    - test_pdb_folder: str, 存放测试集 PDB 文件的目录
    - upper_lower_of_pdb: str='upper'或 'lower'. 控制返回pdb_id的大小写
    输出:
    - test_pdb_ids: set[str], 形状为 (N,), 包含 N 个不重复 PDB ID 字符串的集合
    """
    test_pdb_ids = set()
    for file_name in os.listdir(test_pdb_folder):
        full_path = os.path.join(test_pdb_folder, file_name)
        if not os.path.isfile(full_path):  # 检查是否为文件
            continue
        # pdb_id: str, 标量; 从文件名中分离出的不带后缀的部分
        pdb_id = os.path.splitext(file_name)[0]
        if pdb_id and upper_lower_of_pdb == 'upper':
            test_pdb_ids.add(pdb_id.upper())
        elif pdb_id and upper_lower_of_pdb == 'lower':
            test_pdb_ids.add(pdb_id.lower())
    return test_pdb_ids

def dict_to_list_of_single_dicts(data_dict: dict[str, str]) -> list[dict[str, str]]:
    """
    将包含多个键值对的字典Dict[str, str]转换为包含多个单键值对字典的列表List[Dict[str, str]]
    - data_dict: dict[str, str], 形状为 {k1:v1, k2:v2, ...}, 原始存储 EMDB-PDB 对应关系的字典[{k1:v1}, {k2:v2}, ...]
    输出:
    - list[dict[str, str]]: list, 形状为 (N,), 长度为 N 的列表，每个元素是形如 {key: value} 的小字典
    """
    return [{k: v} for k, v in data_dict.items()]





def make_EMDB_PDB_dict(emdb_to_pdb_csv_path=EMDB_TO_PDB_CSV_PATH, emdb_column_name=EMDB_COLUMN_NAME, pdb_column_name=PDB_COLUMN_NAME, 
                        add_prefix_of_emdb: str='emd_', upper_lower_of_pdb: str='upper', 
                        return_pdb_mode: str='Dict[str, str]', 
                        exclude_pdb_list: list[str] = None):
    """ 返回 EMDB-PDB 字典.返回格式为 return_pdb_mode
    - return_pdb_mode: 'Dict[str, List[str]]' 或 'Dict[str, str]', 对应 {'EMD-611081':['9j37', ...], 'EMD-...':['...']} 和 {'EMD-611081':'9j37', 'EMD-...':'...'}(一个EMDB可能对应多个pdb, 但按照后者的模式, 在返回时只选第一个pdb)
    - prefix_of_emdb: 注意, 总是同时支持关于EMDB_ID的好几种格式(按'-'或'_'划分取最后). 但返回的字典中的key = prefix_of_emdb + emdb_id(数字部分)
    - upper_lower_of_pdb: 'upper' 或 'lower', 控制返回字典中 PDB ID 的大小写
    - exclude_pdb_list: list[str], 需要排除的pdb的文件名, 如9j37. 如果一个EMDB的对应PDB列表与exclude_pdb_list的交集非空, 则不返回该EMDB对应的键值对.注意这是在应用 upper_lower_of_pdb调整了emdb_to_pdb_csv_path之后再检测的.
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


    if return_pdb_mode == 'Dict[str, List[str]]':
        if upper_lower_of_pdb == "upper":
            df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(   # 对每一列(如['9j37', '9j38'])使用
                lambda x: [item.strip().upper() for item in str(x).split(',')]    if ',' in str(x)
                else [str(x).upper()]
            )
        elif upper_lower_of_pdb == "lower":
            df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(   # 对每一列(如['9j37', '9j38'])使用
                lambda x: [item.strip().lower() for item in str(x).split(',')]    if ',' in str(x)
                else [str(x).lower()]
            )
        df_dict['temp_pdbs_list'] = df_dict['temp_pdbs_list'].apply(lambda x: x if set(x).isdisjoint(exclude_pdb_set) else None)  # 过滤

    elif return_pdb_mode == 'Dict[str, str]':
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
    else:
        raise ValueError("return_pdb_mode 格式无法识别")


    processed_dict = dict(zip(df_dict['temp_emdb_id'], df_dict['temp_pdbs_list']))
    filtered_dict = {}
    for emdb_id, pdb in processed_dict.items():
        if pdb is not None:
            filtered_dict[emdb_id] = pdb

    return filtered_dict

def get_existing_emdb_pdb(emdb_to_pdb_dict, 
                          emdb_folder=None, 
                          pdb_folder=None, parsed_pdb_root=None):
    """ 从emdb_to_pdb_dict中筛选条目, 使得所有筛选后的条目在emdb_folder和pdb_folder中都存在.(对于emdb只统一检查数字部分) 
    - emdb_to_pdb_dict: dict[str, str] 或 dict[str, list[str]](按照dict_mode决定, 对应make_EMDB_PDB_dict的return_pdb_mode)
    - emdb_folder: str, emdb文件夹路径
    - pdb_folder: str, pdb文件夹路径
    - parsed_pdb_root: str, parsed_pdb文件夹路径, 由 Pocket_classic\Make_Data\PDB_processor 返回
    """
    emdb_files_set = {os.path.splitext(f)[0].split('_')[-1].split('-')[-1] for f in os.listdir(emdb_folder)}  # 无前后缀, 如'61081'
    pdb_files_set = {os.path.splitext(f)[0] for f in os.listdir(pdb_folder)}  # 无后缀, 如'9j37'
    parsed_pdb_set = {f.upper() for f in os.listdir(parsed_pdb_root)}  # 无后缀, 如'9j37' #NOTE:这里加了upper

    filtered = {}
    for emdb_id, pdb in emdb_to_pdb_dict.items():

        if not isinstance(pdb, list):   # 检查pdb是否存在
            pdb = [pdb]
        if not pdb_files_set.issuperset(set(pdb)):
            continue
        if not parsed_pdb_set.issuperset(set(pdb)):
            continue

        emdb_id_num = emdb_id.split('_')[-1].split('-')[-1]  # 检查emdb_id是否存在
        if emdb_id_num not in emdb_files_set:
            continue

        filtered[emdb_id] = pdb[0] if len(pdb) == 1 else pdb

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
                                            return_pdb_mode=return_pdb_mode, 
                                            exclude_pdb_list = None)  # 可能有些emdb_id对应的pdb不存在, 后面会筛选
    all_emdb_pdb_dict = get_existing_emdb_pdb(all_emdb_pdb_dict, emdb_folder=EMDB_FOLDER, pdb_folder=PDB_FOLDER, parsed_pdb_root=PARSED_PDB_ROOT)  # 无空缺版

    test_pdb_ids = collect_test_pdb_ids(TEST_PDB_FOLDER, upper_lower_of_pdb=upper_lower_of_pdb) #NOTE:注意这里的大写
    test_emdb_pdb_dict = {k: v for k, v in all_emdb_pdb_dict.items() if v in test_pdb_ids}                      # 无空缺+测试集
    test_emdb_pdb_dict = fetch_k_samples(data=test_emdb_pdb_dict, k=MAX_test_num) if MAX_test_num is not None else test_emdb_pdb_dict
    # 依据 MIN_test_ratio 检查并补全测试集 (此时 test_emdb_pdb_dict 中已有预定的测试集数据)
    if MIN_test_ratio is not None:
        target_test_num = int(len(all_emdb_pdb_dict) * MIN_test_ratio)
        current_test_num = len(test_emdb_pdb_dict)
        if current_test_num < target_test_num:
            # 需要从不在当前测试集的剩余样本中另外抽取不足的数量
            remaining_dict = {k: v for k, v in all_emdb_pdb_dict.items() if k not in test_emdb_pdb_dict}
            needed = target_test_num - current_test_num
            extra_test_dict = fetch_k_samples(data=remaining_dict, k=needed)
            test_emdb_pdb_dict.update(extra_test_dict)

    train_val_emdb_pdb_dict = {k: v for k, v in all_emdb_pdb_dict.items() if k not in test_emdb_pdb_dict}       # 无空缺+训练验证集



    # ========== Step 1: 创建全部的.json ==========
    print("\n[Step 1] 创建全部的.json...")
    all_emdb_pdb_json_path = OUTPUT_DIR / "all.json"    # 无空缺版;所有
    save_json(dict_to_list_of_single_dicts(all_emdb_pdb_dict), all_emdb_pdb_json_path)
    

    # ========== Step 2: 创建测试集.json ==========
    print("\n[Step 2] 创建测试集.json...")
    test_emdb_pdb_json_path = OUTPUT_DIR / "test.json"  # 无空缺+测试集
    save_json(dict_to_list_of_single_dicts(test_emdb_pdb_dict), test_emdb_pdb_json_path)


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
    print(f"\n[Step 5] 创建小验证集.json... (n={num_val_parts})")
    list_ = split_into_n_parts(val_emdb_pdb_dict, n=num_val_parts)
    for i, list_i in enumerate(list_):
        path = OUTPUT_DIR / f"val_{i}.json"
        save_json(dict_to_list_of_single_dicts(list_i), path)

    
    # ========== Step 6: 创建小训练集.json ==========
    print(f"\n[Step 6] 创建小训练集.json... (n={num_train_parts})")
    list_ = split_into_n_parts(train_emdb_pdb_dict, n=num_train_parts)
    for i, list_i in enumerate(list_):
        path = OUTPUT_DIR / f"train_{i}.json"
        save_json(dict_to_list_of_single_dicts(list_i), path)
    


    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print(f"完成! 请检查目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 打印汇总
    print("\n划分汇总:")
    print(f"  train.json: {len(train_emdb_pdb_dict)} 样本")
    print(f"  val.json: {len(val_emdb_pdb_dict)} 样本")
    print(f"  test.json: {len(test_emdb_pdb_dict)} 样本")


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
