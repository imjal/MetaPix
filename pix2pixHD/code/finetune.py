import pdb
from code.train import train
from code.test import test
from options.both_options import BothOptions
from code.data_loader import load_dataset, load_train_dataset
import os
import numpy as np
import pickle
import time

def finetune(opt): 
    start_time = time.time()
    if opt.T != None: 
        opt.niter = opt.T

    txtimg_finetune, txtlabel_finetune, txtimg_test, txtlabel_test = load_dataset(opt.test_dataset)

    # set up for all test videos, to finetune on 85% of the the video, then test on the other 15% and compute ssim and metrics
    checkpoint_names = []
    # copy checkpoints_dir 8 times in order to have fresh check updating for each one
    for i in range(8):
        ckpt_i = opt.exp_root_dir + "/checkpoints/test_" + str(i+1)
        os.makedirs(ckpt_i, exist_ok=True)
        checkpoint_names.append(ckpt_i)
        # create viz_directories
        os.makedirs(opt.exp_root_dir + "/viz/" + str(i+1), exist_ok=True)
        os.makedirs(opt.exp_root_dir + "/viz/"+ str(i+1) + "/generated/", exist_ok= True)


    scores = np.zeros((len(txtlabel_finetune), 3))

    os.makedirs(opt.exp_root_dir + "/results", exist_ok=True)
    orig_batch_size = opt.batchSize
    orig_epoch = opt.which_epoch
    # for each video
    for i in range(opt.start_FT_vid-1, len(txtlabel_finetune)):
        print("---Finetuning Video " + str(i))
        # finetuning --continue training option, batchSize 1 for divisibility purposes
        # change options for training
        opt.txtfile_label = txtlabel_finetune[i]
        opt.txtfile_img = txtimg_finetune[i]
        opt.checkpoints_dir = checkpoint_names[i]
        opt.which_epoch = 'latest'
        opt.isTrain = True
        opt.batchSize = orig_batch_size
        opt.which_epoch = orig_epoch
        train(opt)
        time.sleep(5)
        # Inference
        print("---Testing Video " + str(i))
        opt.isTrain = False
        opt.which_epoch = 'latest'
        opt.results_dir = opt.exp_root_dir + "/results/test_" + str(i+1)
        opt.txtfile_label = txtlabel_test[i]
        opt.txtfile_img = txtimg_test[i]
        opt.viz_dir = opt.exp_root_dir + "/viz/" + str(i+1) + "/"
        opt.viz_dir_gen = opt.exp_root_dir + "/viz/"+ str(i+1) + "/generated/"
        scores[i] = test(opt)
        print(scores[i])


    # avg scores across all videos
    
    mean_scores = scores.mean(axis = 0)
    dic = {"scores matrix": scores, "mean": mean_scores}
    with open(opt.exp_root_dir + '/results.pkl', 'wb') as f:
        pickle.dump(dic, f)
    end = time.time()
    print("Execution Time: " + str(end-start_time))
    print("Scores matrix: ")
    print(scores)
    print("Score Means: SSIM, PSNR, MSE")
    print(mean_scores)
    return mean_scores