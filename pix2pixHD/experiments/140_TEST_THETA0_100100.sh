#!/bin/bash
script="$BASH_SOURCE"
echo "Called $script"
python launch.py --mode test_theta_init --test_dataset datasets/testset_5_v3 --niter 10 --niter_decay 10 --batchSize 8 --k 5 --name 140_TEST_THETA0_100100_REDO
