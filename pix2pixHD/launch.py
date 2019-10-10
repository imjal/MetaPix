import argparse
import pdb
from code.train import train
from code.test import test
from options.both_options import BothOptions
from code.data_loader import load_dataset, load_train_dataset
import os
import numpy as np
import pickle
from code.reptile import meta_train
import time
from code.run_test_auto import run_test
from code.finetune import finetune

#add --checkpoints_dir of old model, and --continue_train when you finetune the model
#add --checkpoints_dir for training from scratch
#add --dataroot for training from the data directory that contains train_img and train_label
#add --txtfile_label
#add --txtfile_img
#add --use_only_vgg_loss
#add --name


def make_opt_txt(opt):
    """
    Remnant of the Pix2PixHD codebase
    """
    file_name = os.path.join(opt.checkpoints_dir, 'opt.txt')
    with open(file_name, 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def check_if_delete(opt): 
    while True:
        x = input("Would you like to delete the directory " + opt.exp_root_dir + " [Y/n] \t")
        if x == "Y":
            os.system("rm -rf " + opt.exp_root_dir)
            print("---Finished removing directory, continuing with program")
            os.makedirs(opt.exp_root_dir)
            return True
        elif x == "n":
            while True:
                y = input("Would you like continue, possibly overwriting the directory? [Y/n] \t")
                if y == "Y": 
                    print("---Continue script, with potential of overwriting directory")
                    return True
                elif y == "n": 
                    print("Exiting Launch Program")
                    exit()


def main(opt):
    # creating root directory for experiment 
    opt.exp_root_dir = os.path.join("data-meta/experiments/", opt.name)
    try:
        os.makedirs(opt.exp_root_dir)
    except FileExistsError:
        check_if_delete(opt)
    # sometimes test directories that already have checkpoints
    if opt.mode != 'test_checkpoints' and opt.mode != 'test-one':
        opt.checkpoints_dir = os.path.join(opt.exp_root_dir, "checkpoints")
        os.makedirs(opt.checkpoints_dir, exist_ok = True)
    make_opt_txt(opt)

    # mode specific launching
    if opt.mode == 'train':
        print("--Training " + opt.exp_root_dir)
        A, B = load_train_dataset(opt.train_dataset)
        opt.isTrain = True
        opt.txtfile_label = B[0]
        opt.txtfile_img = A[0]
        train(opt)
        print("[Launcher] Finished Training " + opt.name)
    elif opt.mode == "meta-train":
        if opt.T != None:
            opt.niter = opt.T
        meta_train(opt)
    elif opt.mode == 'finetune':
        finetune(opt)
    elif opt.mode == 'test':
        checkpoints_other = opt.checkpoints_dir
        for i in range(8):
            os.makedirs(opt.exp_root_dir + f"/viz/{i+1}/generated/", exist_ok= True)
        # Inference
        txtimg_finetune, txtlabel_finetune, txtimg_test, txtlabel_test = load_dataset(opt.test_dataset)
        scores = np.zeros((len(txtlabel_finetune), 3))
        os.makedirs(opt.exp_root_dir + "/results", exist_ok=True)
        for i in range(opt.start_FT_vid-1, 8): 
            print("---Testing Video " + str(i))
            opt.checkpoints_dir = os.path.join(checkpoints_other,  f"test_{i+1}")
            opt.isTrain = False
            opt.results_dir = opt.exp_root_dir + f"/results/test_{i+1}"
            opt.txtfile_label = txtlabel_test[i]
            opt.txtfile_img = txtimg_test[i]
            opt.viz_dir = opt.exp_root_dir + f"/viz/{i+1}/"
            opt.viz_dir_gen = opt.exp_root_dir + f"/viz/{i+1}/generated/"
            scores[i] = test(opt)
    elif opt.mode == 'test_checkpoints':
        run_test(opt)
    elif opt.mode == 'test_all_list':
        """
        Test an weights at a certain epoch to observe finetuning
        """
        for j in [10, 20, 30, 40, 50, 80, 100, 200]: 
            txtimg_finetune, txtlabel_finetune, txtimg_test, txtlabel_test = load_dataset(opt.test_dataset)
            scores = np.zeros((len(txtlabel_finetune), 3))
            os.makedirs(opt.exp_root_dir + "/results", exist_ok=True)
            i = opt.one_vid
            os.makedirs(opt.exp_root_dir + f"/{j}/viz/{i}", exist_ok=True)
            os.makedirs(opt.exp_root_dir + f"/{j}/viz/{i}/generated/", exist_ok= True)
            print(f"---Testing Video " + str(opt.one_vid) + f"\t At epoch {j}")
            opt.which_epoch = j;
            opt.isTrain = False
            opt.results_dir = opt.exp_root_dir + f"/results/test_{i}"
            opt.txtfile_label = txtlabel_test[i-1]
            opt.txtfile_img = txtimg_test[i-1]
            opt.viz_dir = opt.exp_root_dir + f"/{j}/viz/{i}/"
            opt.viz_dir_gen = opt.exp_root_dir + f"/{j}/viz/{i}/generated/"
            scores[i] = test(opt)
    elif opt.mode == 'test_theta_init': 
        # takes in any weights, create a directory that can test the weights, and save to same file for testing purposes
        for i in range(8):
            os.makedirs(opt.exp_root_dir + f"/viz/{i+1}/generated/", exist_ok= True)
        if opt.one_vid == 0:
            # Inference
            txtimg_finetune, txtlabel_finetune, txtimg_test, txtlabel_test = load_dataset(opt.test_dataset)
            scores = np.zeros((len(txtlabel_finetune), 3))
            os.makedirs(opt.exp_root_dir + "/results", exist_ok=True)
            for i in range(opt.start_FT_vid-1, 8): 
                print(f"---Testing Video {i+1}")
                opt.isTrain = False
                opt.results_dir = opt.exp_root_dir + f"/results/test_{i+1}"
                opt.txtfile_label = txtlabel_test[i]
                opt.txtfile_img = txtimg_test[i]
                opt.viz_dir = opt.exp_root_dir + f"/viz/{i+1}/"
                opt.viz_dir_gen = opt.exp_root_dir + f"/viz/{i+1}/generated/"
                scores[i] = test(opt)
        else:
            txtimg_finetune, txtlabel_finetune, txtimg_test, txtlabel_test = load_dataset(opt.test_dataset)
            scores = np.zeros((len(txtlabel_finetune), 3))
            os.makedirs(opt.exp_root_dir + "/results", exist_ok=True)
            i = opt.one_vid
            print(f"---Testing Video {opt.one_vid+1}")
            opt.isTrain = False
            opt.results_dir = opt.exp_root_dir + f"/results/test_{i}"
            opt.txtfile_label = txtlabel_test[i-1]
            opt.txtfile_img = txtimg_test[i-1]
            opt.viz_dir = opt.exp_root_dir + f"/viz/{i}/"
            opt.viz_dir_gen = opt.exp_root_dir + f"/viz/{i}/generated/"
            scores[i] = test(opt)
    else: 
        print("not a valid mode, quitting")


if __name__ == "__main__": 
    opt = BothOptions().parse()
    main(opt)