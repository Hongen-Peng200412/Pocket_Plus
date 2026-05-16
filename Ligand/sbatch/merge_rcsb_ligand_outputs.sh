#!/bin/bash
set -eo pipefail

source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
cd /home/penghongen/My_Project/Pocket_Plus

python -m Ligand.rcsb_enrichment.merge_outputs \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-count 5
