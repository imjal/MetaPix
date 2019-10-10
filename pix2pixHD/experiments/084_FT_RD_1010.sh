#/bin/bash
script="$BASH_SOURCE"
echo "Called $script"
python launch.py --mode finetune  --test_dataset datasets/testset_5_v3 --niter 10 --niter_decay 10 --batchSize 8 --k 5 --name 084_FT_RD_1010_REDO

