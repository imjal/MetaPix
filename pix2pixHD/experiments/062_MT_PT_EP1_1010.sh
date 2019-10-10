#
/bin/bash
script="$BASH_SOURCE"
echo "Called $script"
python launch.py --mode meta-train --train_dataset datasets/train_set --test_dataset datasets/testset_5_v3 --k 5 --niter 10 --niter_decay 10 --name 062_MT_PT_EP1_1010_REDO --save_latest_freq 100000 --save_epoch_freq 1000 --meta_iter 24001 --batchSize 8 --dataroot /data/jl5/data-meta/ --tf_log --load_pretrain data-meta/experiments/062_MT_PT_EP1_1010/checkpoints --save_meta_iter 300 --epsilon 1  --start_meta_iter 601
