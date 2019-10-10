#!/bin/bash
script="$BASH_SOURCE"
echo "Called $script"
python launch.py --mode finetune --test_dataset datasets/testset_5_v3 --load_pretrain data-meta/experiments/021_TRAIN_TSV2/checkpoints --which_epoch latest --niter 10 --niter_decay 10 --batchSize 8 --k 5 --name 065_FT_PT_1010_REDO

