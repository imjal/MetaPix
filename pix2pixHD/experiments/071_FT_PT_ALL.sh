#!/bin/bash
script="$BASH_SOURCE"
echo "Called $script"
python launch.py --mode finetune --dataroot /data/jl5/data-meta --test_dataset datasets/testset_split_85_v3  --batchSize 8 --T 5 --name 071_FT_PT_ALL_REDO_1
