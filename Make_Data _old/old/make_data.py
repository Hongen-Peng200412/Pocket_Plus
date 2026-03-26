import numpy as np
import sys
import os
sys.path.append("/home/penghongen")
from Ligand.utils.Make_More_Label import dilate_label
from Ligand.utils.network_tools import atomic_np_savez, save_map_perfect_copy
output_path_prefix = "/storage/penghongen/Pocket/Label_1.5A_with3BOX/Label/"

"""
This code is used to generate pocket labels for ligand in the pocket.
逻辑为：首先对原始label(0-背景  1-主链 2-ligand)中的ligand做5A的球膨胀，若一个主链原子在某个ligand膨胀后获得的球内， 则标记为pocket。
"""
original_ligand_prefix = "/storage/penghongen/Ligand/LABEL_1.5A_BOX_1/0/"  # label: ligand为1其余全0
original_chain_prefix = "/storage/penghongen/Ligand/LABEL_1.5A_BOX_1/2/"   # label: 主链为1其余全0
original_name_list = os.listdir(original_ligand_prefix)  # BOX: emd_44957_11.npz, (grid, x_range, y_range, z_range, voxel_size, origin)

ligand_label_path = [os.path.join(original_ligand_prefix, name) for name in original_name_list] 
chain_label_path = [os.path.join(original_chain_prefix, name) for name in original_name_list] 



# ------------------------------------------------------------ 多CPU时 -----------------------------------------------------
# import sys
# num_tasks = 10
# task_id = int(sys.argv[1])
# assert 0 <= task_id < num_tasks, f"task_id must be in [0, {num_tasks-1}]"
# num_files = len(original_name_list)
# start_index = (num_files // num_tasks) * task_id
# end_index = (num_files // num_tasks) * (task_id + 1) if task_id != num_tasks - 1 else num_files

# # 在一开始加上这个就好了
#     # if count < start_index or count >= end_index:
#     #     continue
# ------------------------------------------------------------ 多CPU时 -----------------------------------------------------

num_all_grid, num_chain_grid, num_pocket_grid, num_ligand_grid = 0, 0, 0, 0
for count, npz_name in enumerate(original_name_list):   # npz_name: emd_44957_11.npz


    # if count < start_index or count >= end_index:
    #     continue



    print(f"--------开始处理文件{original_name_list[count]}-----------------")
    # 接下来，我将会把ligand按球状膨胀
    original_ligand_npz = np.load(ligand_label_path[count])

    ori_ligand_label, x_range, y_range, z_range, voxel_size, origin = original_ligand_npz['grid'], original_ligand_npz['x_range'], original_ligand_npz['y_range'], original_ligand_npz['z_range'], original_ligand_npz['voxel_size'], original_ligand_npz['origin']

    ori_ligand_label = ori_ligand_label.astype(np.int64)

    if not os.path.exists(chain_label_path[count]):
        raise RuntimeError(f"主链标签(hardmask) {chain_label_path[count]} not exists")
    original_chain_npz = np.load(chain_label_path[count])
    ori_chain_label = original_chain_npz['grid']


    output_path = os.path.join(output_path_prefix, npz_name)
    ligand_label_dilated = dilate_label(label=ori_ligand_label, num_classes=2, dilate_size=[1,3.5], has_batch=False, cube_rather_ball=False, 
                         save_path=None, return_tensor_rather_numpy=False)   # 0-背景  1-膨胀后的ligand
    pocket_label = (ligand_label_dilated==1) & (ori_chain_label==1)
    pocket_label = pocket_label.astype(np.int64)
    atomic_np_savez(output_path, grid=pocket_label, x_range=x_range, y_range=y_range, z_range=z_range, voxel_size=voxel_size, origin=origin, do_not_replace=False)


    # if count < 100:
    #     map_output_path = os.path.join("/storage/penghongen/Pocket/Label_1.5A_with3BOX/map/", npz_name[:-4] + ".mrc") # ../emd_44957_11
    #     emdb_id = "emd_" + npz_name.split("_")[1]
    #     origin_map_path = os.path.join("/storage/penghongen/1.5A_map/", emdb_id + ".mrc")

    #     save_map_perfect_copy(file_path=map_output_path, data=pocket_label, original_map_path=origin_map_path, new_origin_xyz=origin)
    

    num_all_grid += np.prod(pocket_label.shape)
    num_chain_grid += np.sum((ori_chain_label == 1).astype(np.int64))
    num_pocket_grid += np.sum((pocket_label == 1).astype(np.int64))
    num_ligand_grid += np.sum((ori_ligand_label == 1).astype(np.int64))
    # print(f"num_all_grid, num_chain_grid, num_pocket_grid, num_ligand_grid: {num_all_grid, num_chain_grid, num_pocket_grid, num_ligand_grid} \n\n ")



print(f" \n\n -----------------全部处理完毕------------------ ")
print(f"num_all_grid, num_chain_grid, num_pocket_grid, num_ligand_grid: {num_all_grid, num_chain_grid, num_pocket_grid, num_ligand_grid} \n\n ")
# 对于“选择指标1（大于2.9A且ligand占比大于2e-4则纳入这个BOX）”切出来的BOX，它们所有的 体素数之和, 包含主链原子的所有体素之和， 包含pocket（距离ligand3.5个格子之内）的所有体素之和， 所有包含ligand的所有体素之和


    
    