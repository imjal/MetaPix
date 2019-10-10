import numpy as np
import os
import glob
from code.test import test
import time
from options.test_options import TestOptions
from code.data_loader import load_dataset, load_train_dataset
from code.finetune import finetune
import shutil
from copy import deepcopy


def get_latest_checkpoint(checkpoints_dir):
    list_of_files = glob.glob(checkpoints_dir + "/*.pth")
    if list_of_files == []:
      return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file

def compare_scores(old, old_ckpt, new, new_ckpt, epsilon):
    print("--called compare_scores")
    # ssim , psnr, mse
    ssim_dif = (new[0] - old[0])
    psnr_dif = (new[1] - old[1])
    mse_dif = (old[2] - new[2])
    print("Score Diff: SSIM, PSNR, MSE ")
    print(ssim_dif, psnr_dif, mse_dif)

    if ssim_dif > 0 or psnr_dif > 0 or mse_dif > 0:
        return new, new_ckpt
    else:
        return old, old_ckpt

def write_scores_file(opt, scores, epoch):
    with open(opt.log_file, "a") as f:
        f.write(str(epoch) + "\t" + str(scores) + "\n")

def run_test(opt):
    # previous test performance
    old_max_score = []
    old_max_ckpt = 0
    prev_checkpoints = []
    old_checkpoints_dir = opt.checkpoints_dir
    opt.log_file = opt.exp_root_dir + "/log_file.txt"
    main_exp_dir = opt.exp_root_dir
    old_opt = deepcopy(opt)
    while(True):
        #check for new checkpoint in directory
        latest = get_latest_checkpoint(old_checkpoints_dir) # some directory to grab from
        if latest == None:
            print("--Sleeping for 5 min from no checkpoints")
            time.sleep(5*60)
            continue
        latest_filename = latest[len(old_checkpoints_dir) + 1:].replace("_net_D.pth", "").replace("_net_G.pth", "")
        if latest in prev_checkpoints:
            print("--sleeping for 5 min")
            time.sleep(5 * 60) # 5 minutes
            continue
        try:
            #test once a directory is there
            opt.exp_root_dir = os.path.join(main_exp_dir, latest_filename)
            opt.which_epoch = latest_filename
            opt.load_pretrain = old_checkpoints_dir
            print("--going into finetuning")
            mean_scores_current = finetune(opt)
            print("--Latest Checked: " + str(latest))
            print("--Current Scores: " + str(mean_scores_current))
        except Exception as e:
            print("--expection occured ")
            #retry & remove from prev_checkpoints
            shutil.rmtree(opt.exp_root_dir)
            opt = deepcopy(old_opt)
            time.sleep(5)
            continue
        write_scores_file(opt, mean_scores_current, latest_filename)
        # old_max_score, old_max_ckpt = compare_scores(old_max_score, old_max_ckpt, mean_scores_current, latest, 0.1)
        prev_checkpoints+=[latest]

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.exp_root_dir = "data-meta/experiments/" + opt.name
    os.makedirs("data-meta/experiments/" + opt.name)
    print(run_test(opt))
