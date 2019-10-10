import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
from models.models import create_model
from code.data_loader import load_dataset, load_train_dataset
from code.train import train
import os
import tensorflow as tf
from code.finetune import finetune
import pdb


def save_meta_weights(checkpoints_dir, new_G_weights, new_D_weights, epoch):
    torch.save(new_G_weights, checkpoints_dir + "/" + ('%s_net_%s.pth' % (epoch, 'G')))
    torch.save(new_D_weights, checkpoints_dir + "/" +  ('%s_net_%s.pth' % (epoch, 'D')))

def plot_current_errors(writer, errors, step, total_loss_dict):
    for tag, value in errors.items():
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)
    for tag, value in total_loss_dict.items():
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)

def plot_testing(writer, mean_scores, step):
    lst = ['SSIM', 'PSNR', 'MSE']
    for i in range(len(lst)): 
        summary = tf.Summary(value=[tf.Summary.Value(tag=lst[i], simple_value=mean_scores[i])])
        writer.add_summary(summary, step)


def meta_train(opt):
    opt.isTrain = True
    # will load pretrained model if --load_pretrain is set to something
    model = create_model(opt)
    G_dict, D_dict = model.module.return_dicts()
    Pre_D_dict = deepcopy(D_dict)
    writer = tf.summary.FileWriter(os.path.join(opt.exp_root_dir, "meta_loss_log"))

    # Reptile training loop
    # will only load the model initial_weights
    opt.load_pretrain = "" 
    txtimg_train, txtlabel_train = load_train_dataset(opt.train_dataset)
    checkpoints_dir = opt.checkpoints_dir
    old_opt = deepcopy(opt)
    old_exp_root = opt.exp_root_dir
    for iteration in range(opt.start_meta_iter, opt.meta_iter):
        opt = deepcopy(old_opt)
        G_dict_before = deepcopy(G_dict)
        D_dict_before = deepcopy(D_dict)

        # Generate task
        idx = np.random.randint(len(txtlabel_train))
        torch.cuda.set_device(0)
        opt.init_weights = True
        opt.txtfile_label = txtlabel_train[idx]
        opt.txtfile_img = txtimg_train[idx]
        opt.checkpoints_dir = old_exp_root + "/checkpoints_" + str(iteration) + "/"
        os.makedirs(opt.checkpoints_dir, exist_ok = True)
        opt.isTrain = True
        opt.load_pretrain = ""
        opt.G_dict = G_dict
        if opt.only_generator:
            print("--loaded pretrained discriminator")
            opt.D_dict = Pre_D_dict
        else:
            opt.D_dict = D_dict

        # I am passing in init_weights to load the latest.pth model weights as initial weights
        # also passing in the dictionaries to load so we don't have to save them
        model, loss_dict, total_loss_dict = train(opt)

        plot_current_errors(writer, loss_dict, iteration, total_loss_dict)
        G_dict, D_dict = model.module.return_dicts() # grab dictionaries
        opt.epsilon = opt.epsilon * (1 - iteration / opt.meta_iter) # linear schedule

        G_dict = {name : G_dict_before[name] + (G_dict[name] - G_dict_before[name]) * opt.epsilon
            for name in G_dict_before}
        D_dict = {name : D_dict_before[name] + (D_dict[name] - D_dict_before[name]) * opt.epsilon
            for name in D_dict_before}

        if iteration % opt.save_meta_iter == 0 and iteration != 0:
            save_meta_weights(checkpoints_dir, G_dict, D_dict, 'latest')
        
        if iteration % opt.test_meta_iter == 0 and iteration != 0: 
            save_meta_weights(checkpoints_dir, G_dict, D_dict, str(iteration))
            # testing
            opt.init_weights = False
            opt.load_pretrain = checkpoints_dir
            opt.which_epoch = 'latest'
            opt.mode = 'finetune'
            opt.init_weights = False
            opt.exp_root_dir = opt.exp_root_dir + "/finetune_" + str(iteration)
            os.makedirs(opt.exp_root_dir, exist_ok=True)
            mean_scores = finetune(opt)
            plot_testing(writer, mean_scores, iteration)

    save_meta_weights(checkpoints_dir, G_dict, D_dict, 'latest')
