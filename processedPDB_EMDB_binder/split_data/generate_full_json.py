"""
划分逻辑:
1. 我们要构造三种数据集, 对应于三个划分文件夹(Split_output_FOLDER)。 每个文件夹含有 "包含.json文件的子文件夹"————它们目前名称将会是：metal_ion, small_molecule, peptide, nucleic(来自 Pocket\processedPDB_EMDB_binder\bind.py 和 Pocket\processedPDB_EMDB_binder\split_and_select_box.py 的保存逻辑) 每个子文件夹对应 "关于这个数据集、针对这种配体的划分".
2. 目前, 文件夹 split_0 对应的数据集是: 只包含所有以"_C"为结尾的样本, 对应所有结合位点的"中心摄影图"; 
         文件夹 split_3 对应的数据集是: 包含: 所有以"_C"为结尾的样本 + 每个结合位点额外随机选取的3个BOX; 
         文件夹 split_all 对应的数据集是: 所有样本.
3. .json格式形如["9e01_3_0_0_0_C", ...], 它是去后缀的文件名. 某个条目在.json列表内当且仅当这个文件同时存在于EMDB_BOX_FOLDER, PDB_Feaure_BOX_FOLDER, PBD_Label_BOX_FOLDER中.

Note:
(1). 实际操作中，就是依据 Instance_Json = ".../all.json" 来遍历/过滤的，也就是说 Instance_Json = ".../all.json" 相当于"最全的emdb_pdb"映射, 不在里面的样本都忽略.
(2). 本脚本产生的.json与 Pocket\Make_Data\split_data\generate_full_json.py 完全对齐.
具体而言, Json_Root_Folder 这个文件夹里面的.json文件完全由 Pocket\Make_Data\split_data\generate_full_json.py 产生.
本脚本将读取 Json_Root_Folder 里面所有的.json文件, 然后在 Split_output_FOLDER 里面每个子文件夹里生成完全相同的.json. 某个BOX被划分到了 Split_output_FOLDER 里面的X.json 当且仅当 这个BOX所在的样本在 Json_Root_Folder 里面的X.json中.
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

# str, Json根目录, 其中包含从Make_Data生成的各式X.json，如train.json, val.json, test.json等, 这些json规定了PDB_ID的归属，本脚本将完全遵照其分配逻辑
Json_Root_Folder = "/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4"
# str, 全部原样本的基础信息json(通常是 Json_Root_Folder 目录下的 all.json)
Instance_Json = os.path.join(Json_Root_Folder, "all.json")

# 各种切好的BOX的存放文件夹, 注意它们都有子文件夹(metal_ion, peptide, nucleic, small_molecule), 将全部遍历处理
# pdb_feature_BOX 已不再离线生成, 改为模型在线 scatter
EMDB_EXP_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_exp_BOX"
EMDB_SIM_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_sim_BOX"
PBD_Label_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_BOX"
LIGAND_DIST_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_BOX"

# list[int], 三种 split_mode 参数集合 （0为主中心，3为中心外加三非中心BOX, -1为全部)
Split_Modes = [0, 1, 2, 3, 4, -1]
# list[str], 指定上方的 split_modes 所对应的生成目标子文件夹名
Split_Mode_Names = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4', 'split_all']
# list[str], 指定上方的 split_modes 所对应的生成目标子文件夹名
Split_output_FOLDER = [
    "/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_0", 
    "/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_1", 
    "/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_2", 
    "/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_3", 
    "/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_4", 
    "/storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_all"
]




# 下面是或许会用到的工具函数
def save_json(data, path: Path) -> None:
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {path.name} ({len(data)} 样本)")







# 新增工具函数
def collect_box_stems_per_class(folder: str) -> dict:
    """
    遍历 BOX 文件夹下的所有类别子文件夹, 按类别分别收集 .npz 文件的去后缀名(stem).
    folder的文件结构: folder/{class_name}/{pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C].npz

    输入:
        - folder: str, BOX 根文件夹路径 (内含 metal_ion/peptide 等类别子文件夹)
    输出:
        - class_stems: dict[str, set[str]], key=类别名(如 'metal_ion'), value=该类别下所有 stem 的集合
    """
    # dict[str, set[str]], key=class_name, value=该类别下所有stem集合
    class_stems = {}
    for class_name in os.listdir(folder):
        # str, 类别子文件夹的完整路径
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        # set[str], 当前类别下所有 stem
        stems = set()
        for fname in os.listdir(class_dir):
            # str, 去掉后缀的文件名, 如 '9e01_3_0_0_0_C'
            stem = os.path.splitext(fname)[0]
            stems.add(stem)
        class_stems[class_name] = stems
    return class_stems


def get_valid_stems_per_class(
    emdb_exp_folder: str,
    emdb_sim_folder: str,
    label_folder: str,
    ligand_dist_folder: str,
) -> dict:
    """
    对四个 BOX 文件夹按类别分别取交集, 返回每个类别下公共合法的 stem 集合.

    输入:
        - emdb_exp_folder:    str, 实验密度图 BOX 根文件夹
        - emdb_sim_folder:    str, 模拟密度图 BOX 根文件夹
        - label_folder:       str, PDB Label BOX 根文件夹
        - ligand_dist_folder: str, Ligand 距离图 BOX 根文件夹
    输出:
        - valid_per_class: dict[str, set[str]], key=类别名(如 metal_ion / random_BOX), value=该类别下四方公共 stem 集合
    """
    # dict[str, set[str]], 四个文件夹各自的分类别 stem 字典
    emdb_exp_cls  = collect_box_stems_per_class(emdb_exp_folder)
    emdb_sim_cls  = collect_box_stems_per_class(emdb_sim_folder)
    label_cls     = collect_box_stems_per_class(label_folder)
    ligand_cls    = collect_box_stems_per_class(ligand_dist_folder)
    # set[str], 四个文件夹中同时出现的类别名
    all_classes = set(emdb_exp_cls.keys()) & set(emdb_sim_cls.keys()) & set(label_cls.keys()) & set(ligand_cls.keys())

    # dict[str, set[str]], 对每个类别取四方交集
    valid_per_class = {}
    for cls in sorted(all_classes):
        # set[str], 当前类别在四个文件夹中均存在的 stem
        valid_per_class[cls] = emdb_exp_cls[cls] & emdb_sim_cls[cls] & label_cls[cls] & ligand_cls[cls]
        print(f"    [{cls}] EXP:{len(emdb_exp_cls[cls])}  SIM:{len(emdb_sim_cls[cls])}  Label:{len(label_cls[cls])}  LigDist:{len(ligand_cls[cls])}  交集:{len(valid_per_class[cls])}")
    return valid_per_class


def get_pdb_id(stem: str) -> str:
    """
    从 BOX 文件名 stem 中提取 pdb_id.
    stem 格式: {pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C], pdb_id 固定为第一个下划线前的部分 (4位小写字母, 如 '9e01').

    输入:
        - stem: str, 如 '9e01_3_0_12_-6_C'
    输出:
        - str, pdb_id, 如 '9e01'
    """
    return stem.split('_')[0]


def get_binding_site(stem: str) -> str:
    """
    从 BOX 文件名 stem 中提取结合位点标识 ({pdb_id}_{inst_id}).
    stem 格式: {pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C], inst_id 是第二个下划线分隔的整数字符串 (如 '3').

    输入:
        - stem: str, 如 '9e01_3_0_12_-6_C'
    输出:
        - str, '{pdb_id}_{inst_id}', 如 '9e01_3'
    """
    # list[str], 按下划线切分
    parts = stem.split('_')
    # str, 结合位点标识 = pdb_id + '_' + inst_id
    return parts[0] + '_' + parts[1]


def select_split_stems(all_stems: list, split_mode: int) -> list:
    """
    根据 split_mode 从 all_stems 中筛选出该数据集类型对应的样本子集.

    输入:
        - all_stems:  list[str], 所有合法 BOX stem 的列表 (已过滤交集+Instance_Json)
        - split_mode: int, 0 = 只含 _C; k(k>=1) = _C + 每结合位点至多k个非中心; -1 = 全部
    输出:
        - selected: list[str], 筛选后的 stem 列表 (split_mode=3 时有随机性)
    """
    if split_mode == -1:
        # list[str], 全部样本, 直接返回副本
        return list(all_stems)

    if split_mode == 0:
        # list[str], 只保留以 '_C' 结尾的中心 BOX
        return [s for s in all_stems if s.endswith('_C')]

    if split_mode >= 1:
        # dict[str, list[str]], 按结合位点分组; key='pdb_id_inst_id', value=stem列表
        site_to_center = {}      # dict[str, list[str]], 中心 BOX
        site_to_nonCenter = {}   # dict[str, list[str]], 非中心 BOX
        for stem in all_stems:
            # str, 当前 stem 的结合位点标识
            site = get_binding_site(stem)
            if stem.endswith('_C'):
                # setdefault(key, default) 它是字典（dict）的一个内置方法，逻辑如下：
                # 如果键（key）已经在字典里了： 它会返回该键对应的值（Value）。
                # 如果键（key）不在字典里： 它会把这个键插入字典，并将它的值设为 default，然后返回这个新创建的 default 值。
                site_to_center.setdefault(site, []).append(stem)
            else:
                site_to_nonCenter.setdefault(site, []).append(stem)

        # list[str], 最终筛选结果
        selected = []
        # 先把全部中心 BOX 加入
        for stems_list in site_to_center.values():
            selected.extend(stems_list)
        # 再对【有中心BOX的】结合位点随机抽取至多 split_mode 个非中心 BOX(没有 _C 的结合位点不在 site_to_center 中, 不会被额外附加非中心 BOX)
        for site, nc_list in site_to_nonCenter.items():
            if site not in site_to_center:
                # 该结合位点无中心 BOX, 跳过（不额外附加）
                continue
            # int, 实际抽取数量 (不足split_mode个则全取)
            k = min(split_mode, len(nc_list))
            selected.extend(random.sample(nc_list, k))
        return selected

    raise ValueError(f"未知 split_mode: {split_mode}, 合法值为 0 / k(k>=1) / -1")




# ============================================================================
# 主函数 / Main Function
# ============================================================================

def main():
    print(f"开始划分，正在依凭{Json_Root_Folder} 进行文件划分归属对齐...")

    # ========== Step 0: 按类别分别取四个BOX文件夹的公共 stem ==========
    print("\n[Step 0] 按类别收集四个BOX文件夹的各自合法 stem（分类别取四方交集）...")
    # dict[str, set[str]], 形状：比如 {"metal_ion": {"9e01_3_0...", ...}, "random_BOX": {...}, ...}
    valid_per_class = get_valid_stems_per_class(
        EMDB_EXP_BOX_FOLDER, EMDB_SIM_BOX_FOLDER, PBD_Label_BOX_FOLDER, LIGAND_DIST_BOX_FOLDER
    )
    # list[str], 包含的所有类别名，例如 ['metal_ion', 'nucleic', 'peptide', 'random_BOX', 'small_molecule']
    all_class_names = sorted(valid_per_class.keys())
    # int, 全部类别合法 stem 总数
    total_valid = sum(len(v) for v in valid_per_class.values())
    print(f"  发现类别: {all_class_names}")
    print(f"  各类别四方交集 stem 总数: {total_valid}")



    # ========== Step 1: 读取外部配置 Json_Root_Folder 中的全部 JSON ==========
    print(f"\n[Step 1] 扫描 Json_Root_Folder ({Json_Root_Folder})，构建全局 PDB 归属映射表...")
    # dict[str, set[str]], 存储外源 json文件名(例如: 'train_0.json'): 其内部所含的不重复pdb_id小写集合
    json_to_pdb_ids = {}
    json_root = Path(Json_Root_Folder)
    for json_file in json_root.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            # list[dict[str, str]], 典型形状如 [{"emd_123": "9j37"}, {"emd_xxx": "5ab1"}, ...]
            data = json.load(f)
        # set[str], 从上面 data 中提取出所有 .values() 作为唯一标识pdb_id的值的集合
        pdb_ids = {list(item.values())[0].lower() for item in data}
        # 将上述集合录入对齐字典, key:str 指向这批ID
        json_to_pdb_ids[json_file.name] = pdb_ids
        print(f"  已从目标配置读取到外部约束: {json_file.name} (含 {len(pdb_ids)} 个不重复的 PDB_ID 样本)")



    # ========== Step 2: 获取全体名单并在 all.json 里过滤 ==========
    # set[str], 获取囊括全体的主集 all.json 中的包含要素集合
    all_pdb_ids = json_to_pdb_ids.get("all.json", set())
    if not all_pdb_ids:
        print("  警告：在根目录下没有找到 all.json 或者内容为空。这可能导致后续交集过滤落空。")
    for cls in all_class_names:
        before = len(valid_per_class[cls])
        # 剔除在 all.json 中未被记录的样本
        valid_per_class[cls] = {s for s in valid_per_class[cls]   if get_pdb_id(s).lower() in all_pdb_ids or get_pdb_id(s).upper() in all_pdb_ids}
        print(f"    [{cls}] 初始根据 all.json 过滤前:{before}  过滤后:{len(valid_per_class[cls])}")



    # ========== Step 3: 对各种split_mode的数据集分别生成 json ==========
    for ds_idx, (split_mode, split_name) in enumerate(zip(Split_Modes, Split_Mode_Names)):
        print(f"\n{'='*60}")
        print(f"[Dataset {ds_idx}] {split_name}  (split_mode={split_mode})")
        print(f"{'='*60}")
        ds_output_dir = Path(Split_output_FOLDER[ds_idx])
        ds_output_dir.mkdir(parents=True, exist_ok=True)
        # dict[str, int], 追踪每个生成的划分文件内写入了多少个stem数目，例如 {'train.json':15152}
        json_to_stem_count = {jname: 0 for jname in json_to_pdb_ids.keys()}

        for cls in all_class_names:
            print(f"\n  -- [{cls}] --")
            cls_output_dir = ds_output_dir / cls
            cls_output_dir.mkdir(parents=True, exist_ok=True)
            # list[str], 对于当前数据集(split_mode)、当前类别(cls) 产生的有效样本的stem(去后缀)
            # random_BOX 不受 split_mode 约束, 始终取全部 stem (等价于 split_mode=-1)
            if cls == 'random_BOX':
                all_cls_stems = select_split_stems(list(valid_per_class[cls]), split_mode=-1)
            else:
                all_cls_stems = select_split_stems(list(valid_per_class[cls]), split_mode)
            
            for json_name, pdb_ids_set in json_to_pdb_ids.items():
                # list[str], 根据 pdb_ids_set 对候选 stem 进行过滤录入
                # 剔除在 all.json 中未被记录的样本
                target_stems = [s for s in all_cls_stems   if get_pdb_id(s).lower() in pdb_ids_set or get_pdb_id(s).upper() in pdb_ids_set]
                save_json(sorted(target_stems), cls_output_dir / json_name)
                json_to_stem_count[json_name] += len(target_stems)

        print(f"\n  [{split_name}] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 {len(all_class_names)} 个类别):")
        for json_name, count in sorted(json_to_stem_count.items()):
            print(f"    {json_name}: 共分配 {count} 个 stem")

    # ========== 全局完成 ==========
    print("\n" + "=" * 60)
    print("全部外部依赖同步划分完成! 请前往以下目录查收生成的对齐 JSON 文件群:")
    for folder in Split_output_FOLDER:
        print(f"  {folder}")
    print("=" * 60)

if __name__ == "__main__":
    main()






# v_1 数据集
"""
(vnegnn) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/processedPDB_EMDB_binder/split_data/generate_full_json.py
开始划分，正在依凭/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/ 进行文件划分归属对齐...

[Step 0] 按类别收集三个BOX文件夹的各自合法 stem（分类别取三方交集）...
    [metal_ion] EMDB:30642  Feature:30642  Label:30642  交集:30642
    [nucleic] EMDB:22  Feature:22  Label:22  交集:22
    [peptide] EMDB:1035  Feature:1035  Label:1035  交集:1035
    [small_molecule] EMDB:90477  Feature:90477  Label:90477  交集:90477
  发现类别: ['metal_ion', 'nucleic', 'peptide', 'small_molecule']
  各类别三方交集 stem 总数: 122176

[Step 1] 扫描 Json_Root_Folder (/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/)，构建全局 PDB 归属映射表...
  已从目标配置读取到外部约束: train_6.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_2.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_9.json (含 26 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_5.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test.json (含 268 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_1.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_4.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_0.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_7.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_8.json (含 26 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_3.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val.json (含 268 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_6.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_2.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_9.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_5.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: all.json (含 2680 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_1.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_4.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train.json (含 2144 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_0.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_7.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_8.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_3.json (含 27 个不重复的 PDB_ID 样本)
    [metal_ion] 初始根据 all.json 过滤前:30642  过滤后:30642
    [nucleic] 初始根据 all.json 过滤前:22  过滤后:22
    [peptide] 初始根据 all.json 过滤前:1035  过滤后:1035
    [small_molecule] 初始根据 all.json 过滤前:90477  过滤后:90477

============================================================
[Dataset 0] split_0  (split_mode=0)
============================================================

  -- [metal_ion] --
  已保存: train_6.json (637 样本)
  已保存: train_2.json (518 样本)
  已保存: val_9.json (33 样本)
  已保存: val_5.json (35 样本)
  已保存: test.json (736 样本)
  已保存: val_1.json (66 样本)
  已保存: val_4.json (67 样本)
  已保存: val_0.json (59 样本)
  已保存: train_7.json (569 样本)
  已保存: val_8.json (65 样本)
  已保存: train_3.json (543 样本)
  已保存: val.json (813 样本)
  已保存: val_6.json (184 样本)
  已保存: val_2.json (81 样本)
  已保存: train_9.json (512 样本)
  已保存: train_5.json (733 样本)
  已保存: all.json (7633 样本)
  已保存: train_1.json (536 样本)
  已保存: train_4.json (742 样本)
  已保存: train.json (6084 样本)
  已保存: train_0.json (789 样本)
  已保存: val_7.json (49 样本)
  已保存: train_8.json (505 样本)
  已保存: val_3.json (174 样本)

  -- [nucleic] --
  已保存: train_6.json (1 样本)
  已保存: train_2.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val.json (0 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: train_5.json (0 样本)
  已保存: all.json (4 样本)
  已保存: train_1.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train.json (4 样本)
  已保存: train_0.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (3 样本)
  已保存: val_3.json (0 样本)

  -- [peptide] --
  已保存: train_6.json (5 样本)
  已保存: train_2.json (22 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (1 样本)
  已保存: test.json (29 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (4 样本)
  已保存: val_0.json (2 样本)
  已保存: train_7.json (20 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (18 样本)
  已保存: val.json (7 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (17 样本)
  已保存: train_5.json (30 样本)
  已保存: all.json (222 样本)
  已保存: train_1.json (12 样本)
  已保存: train_4.json (17 样本)
  已保存: train.json (186 样本)
  已保存: train_0.json (28 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (17 样本)
  已保存: val_3.json (0 样本)

  -- [small_molecule] --
  已保存: train_6.json (1695 样本)
  已保存: train_2.json (1537 样本)
  已保存: val_9.json (194 样本)
  已保存: val_5.json (155 样本)
  已保存: test.json (2137 样本)
  已保存: val_1.json (308 样本)
  已保存: val_4.json (216 样本)
  已保存: val_0.json (242 样本)
  已保存: train_7.json (1842 样本)
  已保存: val_8.json (272 样本)
  已保存: train_3.json (1818 样本)
  已保存: val.json (2429 样本)
  已保存: val_6.json (298 样本)
  已保存: val_2.json (212 样本)
  已保存: train_9.json (1913 样本)
  已保存: train_5.json (2207 样本)
  已保存: all.json (23575 样本)
  已保存: train_1.json (2054 样本)
  已保存: train_4.json (1955 样本)
  已保存: train.json (19009 样本)
  已保存: train_0.json (1760 样本)
  已保存: val_7.json (245 样本)
  已保存: train_8.json (2228 样本)
  已保存: val_3.json (287 样本)

  [split_0] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 31434 个 stem
    test.json: 共分配 2902 个 stem
    train.json: 共分配 25283 个 stem
    train_0.json: 共分配 2577 个 stem
    train_1.json: 共分配 2602 个 stem
    train_2.json: 共分配 2077 个 stem
    train_3.json: 共分配 2379 个 stem
    train_4.json: 共分配 2714 个 stem
    train_5.json: 共分配 2970 个 stem
    train_6.json: 共分配 2338 个 stem
    train_7.json: 共分配 2431 个 stem
    train_8.json: 共分配 2753 个 stem
    train_9.json: 共分配 2442 个 stem
    val.json: 共分配 3249 个 stem
    val_0.json: 共分配 303 个 stem
    val_1.json: 共分配 374 个 stem
    val_2.json: 共分配 293 个 stem
    val_3.json: 共分配 461 个 stem
    val_4.json: 共分配 287 个 stem
    val_5.json: 共分配 191 个 stem
    val_6.json: 共分配 482 个 stem
    val_7.json: 共分配 294 个 stem
    val_8.json: 共分配 337 个 stem
    val_9.json: 共分配 227 个 stem

============================================================
[Dataset 1] split_3  (split_mode=3)
============================================================

  -- [metal_ion] --
  已保存: train_6.json (1996 样本)
  已保存: train_2.json (1596 样本)
  已保存: val_9.json (102 样本)
  已保存: val_5.json (130 样本)
  已保存: test.json (2221 样本)
  已保存: val_1.json (212 样本)
  已保存: val_4.json (230 样本)
  已保存: val_0.json (197 样本)
  已保存: train_7.json (1734 样本)
  已保存: val_8.json (221 样本)
  已保存: train_3.json (1708 样本)
  已保存: val.json (2494 样本)
  已保存: val_6.json (508 样本)
  已保存: val_2.json (242 样本)
  已保存: train_9.json (1602 样本)
  已保存: train_5.json (2261 样本)
  已保存: all.json (23596 样本)
  已保存: train_1.json (1678 样本)
  已保存: train_4.json (2251 样本)
  已保存: train.json (18881 样本)
  已保存: train_0.json (2460 样本)
  已保存: val_7.json (152 样本)
  已保存: train_8.json (1595 样本)
  已保存: val_3.json (500 样本)

  -- [nucleic] --
  已保存: train_6.json (4 样本)
  已保存: train_2.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val.json (0 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: train_5.json (0 样本)
  已保存: all.json (14 样本)
  已保存: train_1.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train.json (14 样本)
  已保存: train_0.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (10 样本)
  已保存: val_3.json (0 样本)

  -- [peptide] --
  已保存: train_6.json (16 样本)
  已保存: train_2.json (70 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (4 样本)
  已保存: test.json (106 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (14 样本)
  已保存: val_0.json (6 样本)
  已保存: train_7.json (70 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (62 样本)
  已保存: val.json (24 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (62 样本)
  已保存: train_5.json (89 样本)
  已保存: all.json (761 样本)
  已保存: train_1.json (40 样本)
  已保存: train_4.json (62 样本)
  已保存: train.json (631 样本)
  已保存: train_0.json (94 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (66 样本)
  已保存: val_3.json (0 样本)

  -- [small_molecule] --
  已保存: train_6.json (5194 样本)
  已保存: train_2.json (4578 样本)
  已保存: val_9.json (564 样本)
  已保存: val_5.json (523 样本)
  已保存: test.json (6384 样本)
  已保存: val_1.json (990 样本)
  已保存: val_4.json (640 样本)
  已保存: val_0.json (687 样本)
  已保存: train_7.json (5509 样本)
  已保存: val_8.json (824 样本)
  已保存: train_3.json (5499 样本)
  已保存: val.json (7364 样本)
  已保存: val_6.json (831 样本)
  已保存: val_2.json (672 样本)
  已保存: train_9.json (5848 样本)
  已保存: train_5.json (6642 样本)
  已保存: all.json (71260 样本)
  已保存: train_1.json (6162 样本)
  已保存: train_4.json (6029 样本)
  已保存: train.json (57512 样本)
  已保存: train_0.json (5198 样本)
  已保存: val_7.json (752 样本)
  已保存: train_8.json (6853 样本)
  已保存: val_3.json (881 样本)

  [split_3] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 95631 个 stem
    test.json: 共分配 8711 个 stem
    train.json: 共分配 77038 个 stem
    train_0.json: 共分配 7752 个 stem
    train_1.json: 共分配 7880 个 stem
    train_2.json: 共分配 6244 个 stem
    train_3.json: 共分配 7269 个 stem
    train_4.json: 共分配 8342 个 stem
    train_5.json: 共分配 8992 个 stem
    train_6.json: 共分配 7210 个 stem
    train_7.json: 共分配 7313 个 stem
    train_8.json: 共分配 8524 个 stem
    train_9.json: 共分配 7512 个 stem
    val.json: 共分配 9882 个 stem
    val_0.json: 共分配 890 个 stem
    val_1.json: 共分配 1202 个 stem
    val_2.json: 共分配 914 个 stem
    val_3.json: 共分配 1381 个 stem
    val_4.json: 共分配 884 个 stem
    val_5.json: 共分配 657 个 stem
    val_6.json: 共分配 1339 个 stem
    val_7.json: 共分配 904 个 stem
    val_8.json: 共分配 1045 个 stem
    val_9.json: 共分配 666 个 stem

============================================================
[Dataset 2] split_all  (split_mode=-1)
============================================================

  -- [metal_ion] --
  已保存: train_6.json (2523 样本)
  已保存: train_2.json (2150 样本)
  已保存: val_9.json (147 样本)
  已保存: val_5.json (186 样本)
  已保存: test.json (2855 样本)
  已保存: val_1.json (290 样本)
  已保存: val_4.json (305 样本)
  已保存: val_0.json (282 样本)
  已保存: train_7.json (2304 样本)
  已保存: val_8.json (307 样本)
  已保存: train_3.json (2212 样本)
  已保存: val.json (3239 样本)
  已保存: val_6.json (625 样本)
  已保存: val_2.json (296 样本)
  已保存: train_9.json (2125 样本)
  已保存: train_5.json (2888 样本)
  已保存: all.json (30642 样本)
  已保存: train_1.json (2206 样本)
  已保存: train_4.json (2868 样本)
  已保存: train.json (24548 样本)
  已保存: train_0.json (3125 样本)
  已保存: val_7.json (210 样本)
  已保存: train_8.json (2147 样本)
  已保存: val_3.json (591 样本)

  -- [nucleic] --
  已保存: train_6.json (8 样本)
  已保存: train_2.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val.json (0 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: train_5.json (0 样本)
  已保存: all.json (22 样本)
  已保存: train_1.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train.json (22 样本)
  已保存: train_0.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (14 样本)
  已保存: val_3.json (0 样本)

  -- [peptide] --
  已保存: train_6.json (20 样本)
  已保存: train_2.json (88 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (8 样本)
  已保存: test.json (146 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (16 样本)
  已保存: val_0.json (10 样本)
  已保存: train_7.json (88 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (82 样本)
  已保存: val.json (34 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (88 样本)
  已保存: train_5.json (110 样本)
  已保存: all.json (1035 样本)
  已保存: train_1.json (54 样本)
  已保存: train_4.json (90 样本)
  已保存: train.json (855 样本)
  已保存: train_0.json (135 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (100 样本)
  已保存: val_3.json (0 样本)

  -- [small_molecule] --
  已保存: train_6.json (6600 样本)
  已保存: train_2.json (5756 样本)
  已保存: val_9.json (695 样本)
  已保存: val_5.json (713 样本)
  已保存: test.json (8052 样本)
  已保存: val_1.json (1312 样本)
  已保存: val_4.json (771 样本)
  已保存: val_0.json (857 样本)
  已保存: train_7.json (7035 样本)
  已保存: val_8.json (1043 样本)
  已保存: train_3.json (6990 样本)
  已保存: val.json (9354 样本)
  已保存: val_6.json (1030 样本)
  已保存: val_2.json (833 样本)
  已保存: train_9.json (7511 样本)
  已保存: train_5.json (8402 样本)
  已保存: all.json (90477 样本)
  已保存: train_1.json (7837 样本)
  已保存: train_4.json (7660 样本)
  已保存: train.json (73071 样本)
  已保存: train_0.json (6493 样本)
  已保存: val_7.json (961 样本)
  已保存: train_8.json (8787 样本)
  已保存: val_3.json (1139 样本)

  [split_all] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 122176 个 stem
    test.json: 共分配 11053 个 stem
    train.json: 共分配 98496 个 stem
    train_0.json: 共分配 9753 个 stem
    train_1.json: 共分配 10097 个 stem
    train_2.json: 共分配 7994 个 stem
    train_3.json: 共分配 9284 个 stem
    train_4.json: 共分配 10618 个 stem
    train_5.json: 共分配 11400 个 stem
    train_6.json: 共分配 9151 个 stem
    train_7.json: 共分配 9427 个 stem
    train_8.json: 共分配 11048 个 stem
    train_9.json: 共分配 9724 个 stem
    val.json: 共分配 12627 个 stem
    val_0.json: 共分配 1149 个 stem
    val_1.json: 共分配 1602 个 stem
    val_2.json: 共分配 1129 个 stem
    val_3.json: 共分配 1730 个 stem
    val_4.json: 共分配 1092 个 stem
    val_5.json: 共分配 907 个 stem
    val_6.json: 共分配 1655 个 stem
    val_7.json: 共分配 1171 个 stem
    val_8.json: 共分配 1350 个 stem
    val_9.json: 共分配 842 个 stem

============================================================
全部外部依赖同步划分完成! 请前往以下目录查收生成的对齐 JSON 文件群:
  /storage/penghongen/Pocket_classic/v_1/split/split_0
  /storage/penghongen/Pocket_classic/v_1/split/split_3
  /storage/penghongen/Pocket_classic/v_1/split/split_all
============================================================
(vnegnn) [penghongen@master ~]$ 
"""



# v2_raw4_10A(其余7个平行数据集也都跑了, 就是太长没粘贴)
"""
conda activate Pocket_Plus_centos7_cu121_allgpu
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/split_data/generate_full_json.py
(base) [penghongen@master ~]$ conda activate Pocket_Plus_centos7_cu121_allgpu
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/split_data/generate_full_json.py
开始划分，正在依凭/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4 进行文件划分归属对齐...

[Step 0] 按类别收集四个BOX文件夹的各自合法 stem（分类别取四方交集）...
    [metal_ion] EXP:60236  SIM:60236  Label:60236  LigDist:60236  交集:60236
    [nucleic] EXP:16  SIM:16  Label:16  LigDist:16  交集:16
    [peptide] EXP:1560  SIM:1560  Label:1560  LigDist:1560  交集:1560
    [random_BOX] EXP:103207  SIM:103207  Label:103207  LigDist:103207  交集:103207
    [small_molecule] EXP:205848  SIM:205848  Label:205848  LigDist:205848  交集:205848
  发现类别: ['metal_ion', 'nucleic', 'peptide', 'random_BOX', 'small_molecule']
  各类别四方交集 stem 总数: 370867

[Step 1] 扫描 Json_Root_Folder (/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4)，构建全局 PDB 归属映射表...
  已从目标配置读取到外部约束: test_2.json (含 269 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_010.json (含 415 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_025.json (含 1038 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test_3.json (含 54 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test_1.json (含 269 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test_5.json (含 54 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_050.json (含 2076 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test.json (含 969 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train.json (含 4152 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val.json (含 269 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test_0.json (含 269 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: all.json (含 5390 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test_4.json (含 54 个不重复的 PDB_ID 样本)
    [metal_ion] 初始根据 all.json 过滤前:60236  过滤后:60236
    [nucleic] 初始根据 all.json 过滤前:16  过滤后:16
    [peptide] 初始根据 all.json 过滤前:1560  过滤后:1560
    [random_BOX] 初始根据 all.json 过滤前:103207  过滤后:103207
    [small_molecule] 初始根据 all.json 过滤前:205848  过滤后:205848

============================================================
[Dataset 0] split_0  (split_mode=0)
============================================================

  -- [metal_ion] --
  已保存: test_2.json (18 样本)
  已保存: train_010.json (1491 样本)
  已保存: train_025.json (3287 样本)
  已保存: test_3.json (290 样本)
  已保存: test_1.json (394 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (6188 样本)
  已保存: test.json (1442 样本)
  已保存: train.json (12836 样本)
  已保存: val.json (800 样本)
  已保存: test_0.json (577 样本)
  已保存: all.json (15078 样本)
  已保存: test_4.json (163 样本)

  -- [nucleic] --
  已保存: test_2.json (0 样本)
  已保存: train_010.json (0 样本)
  已保存: train_025.json (1 样本)
  已保存: test_3.json (0 样本)
  已保存: test_1.json (0 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (2 样本)
  已保存: test.json (0 样本)
  已保存: train.json (2 样本)
  已保存: val.json (0 样本)
  已保存: test_0.json (0 样本)
  已保存: all.json (2 样本)
  已保存: test_4.json (0 样本)

  -- [peptide] --
  已保存: test_2.json (24 样本)
  已保存: train_010.json (20 样本)
  已保存: train_025.json (91 样本)
  已保存: test_3.json (2 样本)
  已保存: test_1.json (9 样本)
  已保存: test_5.json (1 样本)
  已保存: train_050.json (133 样本)
  已保存: test.json (48 样本)
  已保存: train.json (268 样本)
  已保存: val.json (7 样本)
  已保存: test_0.json (12 样本)
  已保存: all.json (323 样本)
  已保存: test_4.json (0 样本)

  -- [random_BOX] --
  已保存: test_2.json (5248 样本)
  已保存: train_010.json (7739 样本)
  已保存: train_025.json (19841 样本)
  已保存: test_3.json (1000 样本)
  已保存: test_1.json (5251 样本)
  已保存: test_5.json (973 样本)
  已保存: train_050.json (39868 样本)
  已保存: test.json (18609 样本)
  已保存: train.json (79400 样本)
  已保存: val.json (5198 样本)
  已保存: test_0.json (5095 样本)
  已保存: all.json (103207 样本)
  已保存: test_4.json (1042 样本)

  -- [small_molecule] --
  已保存: test_2.json (1281 样本)
  已保存: train_010.json (5026 样本)
  已保存: train_025.json (11954 样本)
  已保存: test_3.json (103 样本)
  已保存: test_1.json (1155 样本)
  已保存: test_5.json (157 样本)
  已保存: train_050.json (24219 样本)
  已保存: test.json (5768 样本)
  已保存: train.json (47611 样本)
  已保存: val.json (2927 样本)
  已保存: test_0.json (3018 样本)
  已保存: all.json (56306 样本)
  已保存: test_4.json (54 样本)

  [split_0] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 5 个类别):
    all.json: 共分配 174916 个 stem
    test.json: 共分配 25867 个 stem
    test_0.json: 共分配 8702 个 stem
    test_1.json: 共分配 6809 个 stem
    test_2.json: 共分配 6571 个 stem
    test_3.json: 共分配 1395 个 stem
    test_4.json: 共分配 1259 个 stem
    test_5.json: 共分配 1131 个 stem
    train.json: 共分配 140117 个 stem
    train_010.json: 共分配 14276 个 stem
    train_025.json: 共分配 35174 个 stem
    train_050.json: 共分配 70410 个 stem
    val.json: 共分配 8932 个 stem

============================================================
[Dataset 1] split_1  (split_mode=1)
============================================================

  -- [metal_ion] --
  已保存: test_2.json (36 样本)
  已保存: train_010.json (2781 样本)
  已保存: train_025.json (6132 样本)
  已保存: test_3.json (531 样本)
  已保存: test_1.json (749 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (11635 样本)
  已保存: test.json (2714 样本)
  已保存: train.json (24046 样本)
  已保存: val.json (1479 样本)
  已保存: test_0.json (1089 样本)
  已保存: all.json (28239 样本)
  已保存: test_4.json (309 样本)

  -- [nucleic] --
  已保存: test_2.json (0 样本)
  已保存: train_010.json (0 样本)
  已保存: train_025.json (2 样本)
  已保存: test_3.json (0 样本)
  已保存: test_1.json (0 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (4 样本)
  已保存: test.json (0 样本)
  已保存: train.json (4 样本)
  已保存: val.json (0 样本)
  已保存: test_0.json (0 样本)
  已保存: all.json (4 样本)
  已保存: test_4.json (0 样本)

  -- [peptide] --
  已保存: test_2.json (44 样本)
  已保存: train_010.json (40 样本)
  已保存: train_025.json (176 样本)
  已保存: test_3.json (4 样本)
  已保存: test_1.json (18 样本)
  已保存: test_5.json (2 样本)
  已保存: train_050.json (262 样本)
  已保存: test.json (92 样本)
  已保存: train.json (525 样本)
  已保存: val.json (14 样本)
  已保存: test_0.json (24 样本)
  已保存: all.json (631 样本)
  已保存: test_4.json (0 样本)

  -- [random_BOX] --
  已保存: test_2.json (5248 样本)
  已保存: train_010.json (7739 样本)
  已保存: train_025.json (19841 样本)
  已保存: test_3.json (1000 样本)
  已保存: test_1.json (5251 样本)
  已保存: test_5.json (973 样本)
  已保存: train_050.json (39868 样本)
  已保存: test.json (18609 样本)
  已保存: train.json (79400 样本)
  已保存: val.json (5198 样本)
  已保存: test_0.json (5095 样本)
  已保存: all.json (103207 样本)
  已保存: test_4.json (1042 样本)

  -- [small_molecule] --
  已保存: test_2.json (2464 样本)
  已保存: train_010.json (9296 样本)
  已保存: train_025.json (22007 样本)
  已保存: test_3.json (202 样本)
  已保存: test_1.json (2215 样本)
  已保存: test_5.json (297 样本)
  已保存: train_050.json (44665 样本)
  已保存: test.json (10852 样本)
  已保存: train.json (87782 样本)
  已保存: val.json (5397 样本)
  已保存: test_0.json (5566 样本)
  已保存: all.json (104031 样本)
  已保存: test_4.json (108 样本)

  [split_1] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 5 个类别):
    all.json: 共分配 236112 个 stem
    test.json: 共分配 32267 个 stem
    test_0.json: 共分配 11774 个 stem
    test_1.json: 共分配 8233 个 stem
    test_2.json: 共分配 7792 个 stem
    test_3.json: 共分配 1737 个 stem
    test_4.json: 共分配 1459 个 stem
    test_5.json: 共分配 1272 个 stem
    train.json: 共分配 191757 个 stem
    train_010.json: 共分配 19856 个 stem
    train_025.json: 共分配 48158 个 stem
    train_050.json: 共分配 96434 个 stem
    val.json: 共分配 12088 个 stem

============================================================
[Dataset 2] split_2  (split_mode=2)
============================================================

  -- [metal_ion] --
  已保存: test_2.json (51 样本)
  已保存: train_010.json (3686 样本)
  已保存: train_025.json (8140 样本)
  已保存: test_3.json (693 样本)
  已保存: test_1.json (1031 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (15542 样本)
  已保存: test.json (3676 样本)
  已保存: train.json (32071 样本)
  已保存: val.json (1949 样本)
  已保存: test_0.json (1483 样本)
  已保存: all.json (37696 样本)
  已保存: test_4.json (418 样本)

  -- [nucleic] --
  已保存: test_2.json (0 样本)
  已保存: train_010.json (0 样本)
  已保存: train_025.json (3 样本)
  已保存: test_3.json (0 样本)
  已保存: test_1.json (0 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (6 样本)
  已保存: test.json (0 样本)
  已保存: train.json (6 样本)
  已保存: val.json (0 样本)
  已保存: test_0.json (0 样本)
  已保存: all.json (6 样本)
  已保存: test_4.json (0 样本)

  -- [peptide] --
  已保存: test_2.json (59 样本)
  已保存: train_010.json (59 样本)
  已保存: train_025.json (239 样本)
  已保存: test_3.json (6 样本)
  已保存: test_1.json (26 样本)
  已保存: test_5.json (3 样本)
  已保存: train_050.json (365 样本)
  已保存: test.json (128 样本)
  已保存: train.json (731 样本)
  已保存: val.json (18 样本)
  已保存: test_0.json (34 样本)
  已保存: all.json (877 样本)
  已保存: test_4.json (0 样本)

  -- [random_BOX] --
  已保存: test_2.json (5248 样本)
  已保存: train_010.json (7739 样本)
  已保存: train_025.json (19841 样本)
  已保存: test_3.json (1000 样本)
  已保存: test_1.json (5251 样本)
  已保存: test_5.json (973 样本)
  已保存: train_050.json (39868 样本)
  已保存: test.json (18609 样本)
  已保存: train.json (79400 样本)
  已保存: val.json (5198 样本)
  已保存: test_0.json (5095 样本)
  已保存: all.json (103207 样本)
  已保存: test_4.json (1042 样本)

  -- [small_molecule] --
  已保存: test_2.json (3344 样本)
  已保存: train_010.json (12335 样本)
  已保存: train_025.json (29076 样本)
  已保存: test_3.json (282 样本)
  已保存: test_1.json (3028 样本)
  已保存: test_5.json (410 样本)
  已保存: train_050.json (58900 样本)
  已保存: test.json (14504 样本)
  已保存: train.json (115927 样本)
  已保存: val.json (7117 样本)
  已保存: test_0.json (7287 样本)
  已保存: all.json (137548 样本)
  已保存: test_4.json (153 样本)

  [split_2] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 5 个类别):
    all.json: 共分配 279334 个 stem
    test.json: 共分配 36917 个 stem
    test_0.json: 共分配 13899 个 stem
    test_1.json: 共分配 9336 个 stem
    test_2.json: 共分配 8702 个 stem
    test_3.json: 共分配 1981 个 stem
    test_4.json: 共分配 1613 个 stem
    test_5.json: 共分配 1386 个 stem
    train.json: 共分配 228135 个 stem
    train_010.json: 共分配 23819 个 stem
    train_025.json: 共分配 57299 个 stem
    train_050.json: 共分配 114681 个 stem
    val.json: 共分配 14282 个 stem

============================================================
[Dataset 3] split_3  (split_mode=3)
============================================================

  -- [metal_ion] --
  已保存: test_2.json (66 样本)
  已保存: train_010.json (4456 样本)
  已保存: train_025.json (9888 样本)
  已保存: test_3.json (828 样本)
  已保存: test_1.json (1281 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (18947 样本)
  已保存: test.json (4526 样本)
  已保存: train.json (39113 样本)
  已保存: val.json (2357 样本)
  已保存: test_0.json (1830 样本)
  已保存: all.json (45996 样本)
  已保存: test_4.json (521 样本)

  -- [nucleic] --
  已保存: test_2.json (0 样本)
  已保存: train_010.json (0 样本)
  已保存: train_025.json (4 样本)
  已保存: test_3.json (0 样本)
  已保存: test_1.json (0 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (8 样本)
  已保存: test.json (0 样本)
  已保存: train.json (8 样本)
  已保存: val.json (0 样本)
  已保存: test_0.json (0 样本)
  已保存: all.json (8 样本)
  已保存: test_4.json (0 样本)

  -- [peptide] --
  已保存: test_2.json (74 样本)
  已保存: train_010.json (78 样本)
  已保存: train_025.json (299 样本)
  已保存: test_3.json (8 样本)
  已保存: test_1.json (34 样本)
  已保存: test_5.json (4 样本)
  已保存: train_050.json (462 样本)
  已保存: test.json (164 样本)
  已保存: train.json (925 样本)
  已保存: val.json (22 样本)
  已保存: test_0.json (44 样本)
  已保存: all.json (1111 样本)
  已保存: test_4.json (0 样本)

  -- [random_BOX] --
  已保存: test_2.json (5248 样本)
  已保存: train_010.json (7739 样本)
  已保存: train_025.json (19841 样本)
  已保存: test_3.json (1000 样本)
  已保存: test_1.json (5251 样本)
  已保存: test_5.json (973 样本)
  已保存: train_050.json (39868 样本)
  已保存: test.json (18609 样本)
  已保存: train.json (79400 样本)
  已保存: val.json (5198 样本)
  已保存: test_0.json (5095 样本)
  已保存: all.json (103207 样本)
  已保存: test_4.json (1042 样本)

  -- [small_molecule] --
  已保存: test_2.json (4153 样本)
  已保存: train_010.json (14843 样本)
  已保存: train_025.json (34800 样本)
  已保存: test_3.json (357 样本)
  已保存: test_1.json (3780 样本)
  已保存: test_5.json (509 样本)
  已保存: train_050.json (70342 样本)
  已保存: test.json (17652 样本)
  已保存: train.json (138631 样本)
  已保存: val.json (8501 样本)
  已保存: test_0.json (8659 样本)
  已保存: all.json (164784 样本)
  已保存: test_4.json (194 样本)

  [split_3] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 5 个类别):
    all.json: 共分配 315106 个 stem
    test.json: 共分配 40951 个 stem
    test_0.json: 共分配 15628 个 stem
    test_1.json: 共分配 10346 个 stem
    test_2.json: 共分配 9541 个 stem
    test_3.json: 共分配 2193 个 stem
    test_4.json: 共分配 1757 个 stem
    test_5.json: 共分配 1486 个 stem
    train.json: 共分配 258077 个 stem
    train_010.json: 共分配 27116 个 stem
    train_025.json: 共分配 64832 个 stem
    train_050.json: 共分配 129627 个 stem
    val.json: 共分配 16078 个 stem

============================================================
[Dataset 4] split_4  (split_mode=4)
============================================================

  -- [metal_ion] --
  已保存: test_2.json (74 样本)
  已保存: train_010.json (4875 样本)
  已保存: train_025.json (10835 样本)
  已保存: test_3.json (904 样本)
  已保存: test_1.json (1428 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (20819 样本)
  已保存: test.json (5006 样本)
  已保存: train.json (42981 样本)
  已保存: val.json (2582 样本)
  已保存: test_0.json (2027 样本)
  已保存: all.json (50569 样本)
  已保存: test_4.json (573 样本)

  -- [nucleic] --
  已保存: test_2.json (0 样本)
  已保存: train_010.json (0 样本)
  已保存: train_025.json (5 样本)
  已保存: test_3.json (0 样本)
  已保存: test_1.json (0 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (10 样本)
  已保存: test.json (0 样本)
  已保存: train.json (10 样本)
  已保存: val.json (0 样本)
  已保存: test_0.json (0 样本)
  已保存: all.json (10 样本)
  已保存: test_4.json (0 样本)

  -- [peptide] --
  已保存: test_2.json (82 样本)
  已保存: train_010.json (88 样本)
  已保存: train_025.json (329 样本)
  已保存: test_3.json (9 样本)
  已保存: test_1.json (38 样本)
  已保存: test_5.json (5 样本)
  已保存: train_050.json (517 样本)
  已保存: test.json (181 样本)
  已保存: train.json (1034 样本)
  已保存: val.json (24 样本)
  已保存: test_0.json (47 样本)
  已保存: all.json (1239 样本)
  已保存: test_4.json (0 样本)

  -- [random_BOX] --
  已保存: test_2.json (5248 样本)
  已保存: train_010.json (7739 样本)
  已保存: train_025.json (19841 样本)
  已保存: test_3.json (1000 样本)
  已保存: test_1.json (5251 样本)
  已保存: test_5.json (973 样本)
  已保存: train_050.json (39868 样本)
  已保存: test.json (18609 样本)
  已保存: train.json (79400 样本)
  已保存: val.json (5198 样本)
  已保存: test_0.json (5095 样本)
  已保存: all.json (103207 样本)
  已保存: test_4.json (1042 样本)

  -- [small_molecule] --
  已保存: test_2.json (4636 样本)
  已保存: train_010.json (16271 样本)
  已保存: train_025.json (37929 样本)
  已保存: test_3.json (399 样本)
  已保存: test_1.json (4232 样本)
  已保存: test_5.json (570 样本)
  已保存: train_050.json (76645 样本)
  已保存: test.json (19470 样本)
  已保存: train.json (151091 样本)
  已保存: val.json (9274 样本)
  已保存: test_0.json (9413 样本)
  已保存: all.json (179835 样本)
  已保存: test_4.json (220 样本)

  [split_4] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 5 个类别):
    all.json: 共分配 334860 个 stem
    test.json: 共分配 43266 个 stem
    test_0.json: 共分配 16582 个 stem
    test_1.json: 共分配 10949 个 stem
    test_2.json: 共分配 10040 个 stem
    test_3.json: 共分配 2312 个 stem
    test_4.json: 共分配 1835 个 stem
    test_5.json: 共分配 1548 个 stem
    train.json: 共分配 274516 个 stem
    train_010.json: 共分配 28973 个 stem
    train_025.json: 共分配 68939 个 stem
    train_050.json: 共分配 137859 个 stem
    val.json: 共分配 17078 个 stem

============================================================
[Dataset 5] split_all  (split_mode=-1)
============================================================

  -- [metal_ion] --
  已保存: test_2.json (98 样本)
  已保存: train_010.json (5720 样本)
  已保存: train_025.json (12863 样本)
  已保存: test_3.json (1073 样本)
  已保存: test_1.json (1744 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (24756 样本)
  已保存: test.json (6041 样本)
  已保存: train.json (51097 样本)
  已保存: val.json (3098 样本)
  已保存: test_0.json (2435 样本)
  已保存: all.json (60236 样本)
  已保存: test_4.json (691 样本)

  -- [nucleic] --
  已保存: test_2.json (0 样本)
  已保存: train_010.json (0 样本)
  已保存: train_025.json (8 样本)
  已保存: test_3.json (0 样本)
  已保存: test_1.json (0 样本)
  已保存: test_5.json (0 样本)
  已保存: train_050.json (16 样本)
  已保存: test.json (0 样本)
  已保存: train.json (16 样本)
  已保存: val.json (0 样本)
  已保存: test_0.json (0 样本)
  已保存: all.json (16 样本)
  已保存: test_4.json (0 样本)

  -- [peptide] --
  已保存: test_2.json (102 样本)
  已保存: train_010.json (114 样本)
  已保存: train_025.json (403 样本)
  已保存: test_3.json (12 样本)
  已保存: test_1.json (48 样本)
  已保存: test_5.json (8 样本)
  已保存: train_050.json (652 样本)
  已保存: test.json (224 样本)
  已保存: train.json (1306 样本)
  已保存: val.json (30 样本)
  已保存: test_0.json (54 样本)
  已保存: all.json (1560 样本)
  已保存: test_4.json (0 样本)

  -- [random_BOX] --
  已保存: test_2.json (5248 样本)
  已保存: train_010.json (7739 样本)
  已保存: train_025.json (19841 样本)
  已保存: test_3.json (1000 样本)
  已保存: test_1.json (5251 样本)
  已保存: test_5.json (973 样本)
  已保存: train_050.json (39868 样本)
  已保存: test.json (18609 样本)
  已保存: train.json (79400 样本)
  已保存: val.json (5198 样本)
  已保存: test_0.json (5095 样本)
  已保存: all.json (103207 样本)
  已保存: test_4.json (1042 样本)

  -- [small_molecule] --
  已保存: test_2.json (5689 样本)
  已保存: train_010.json (18733 样本)
  已保存: train_025.json (43200 样本)
  已保存: test_3.json (508 样本)
  已保存: test_1.json (5196 样本)
  已保存: test_5.json (727 样本)
  已保存: train_050.json (87215 样本)
  已保存: test.json (23110 样本)
  已保存: train.json (172134 样本)
  已保存: val.json (10604 样本)
  已保存: test_0.json (10705 样本)
  已保存: all.json (205848 样本)
  已保存: test_4.json (285 样本)

  [split_all] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 5 个类别):
    all.json: 共分配 370867 个 stem
    test.json: 共分配 47984 个 stem
    test_0.json: 共分配 18289 个 stem
    test_1.json: 共分配 12239 个 stem
    test_2.json: 共分配 11137 个 stem
    test_3.json: 共分配 2593 个 stem
    test_4.json: 共分配 2018 个 stem
    test_5.json: 共分配 1708 个 stem
    train.json: 共分配 303953 个 stem
    train_010.json: 共分配 32306 个 stem
    train_025.json: 共分配 76315 个 stem
    train_050.json: 共分配 152507 个 stem
    val.json: 共分配 18930 个 stem

============================================================
全部外部依赖同步划分完成! 请前往以下目录查收生成的对齐 JSON 文件群:
  /storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_0
  /storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_1
  /storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_2
  /storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_3
  /storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_4
  /storage/penghongen/Pocket_classic/v2_raw4_10A/split/split_all
============================================================
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ 
"""
