这个文件夹只能用于经典方法_classic， 这个文件夹是在 Pocket_classic\Make_Data\PDB_processor 的基础上做处理的. 必须先按照这个文件夹下的 readme.md跑完所有程序, 再看这个文件夹.

要跑这个文件夹，必须：
1.先跑 bind.py 生成一系列的 npz 文件(PDB体素化特征、PDB生成的标签、重采样的map图)，保存.
2.然后再跑 split_and_select_box.py, 在上面生成的文件的基础上进行切块.