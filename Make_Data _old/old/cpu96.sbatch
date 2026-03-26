#!/bin/bash
#SBATCH -o /home/penghongen/Pocket/Make_Data/out.txt
#SBATCH -e /home/penghongen/Pocket/Make_Data/error.txt
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J 1.5A_Label
#SBATCH -n 1


# 运行 Python 脚本，传递必要的参数
/home/penghongen/anaconda3/envs/py310_tensorboard/bin/python /home/penghongen/Pocket/Make_Data/make_data.py