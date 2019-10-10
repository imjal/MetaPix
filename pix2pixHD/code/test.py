### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import pdb 
import pickle
from skimage.measure import compare_ssim, compare_psnr, compare_mse
import numpy as np
import cv2
from PIL import Image
from code.data_loader import choose_k_images

def save_img(visuals, opt, img_path): 
    im1 = Image.fromarray(visuals['synthesized_image'])
    im1.save(opt.viz_dir_gen + img_path[-12:])

def save_scores(scores, data, generated, i):
    A = ((data["image"][0].transpose(0,2).cpu().detach().numpy()) + 1)*128
    B = ((generated[0].transpose(0,2).cpu().detach().numpy()) + 1)*128
    scores[i][0] += compare_ssim(A, B, multichannel=True, data_range=256)
    scores[i][1] += compare_psnr(A, B, data_range=256)
    scores[i][2] += compare_mse(A, B)
    return scores

def test(opt):
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.no_instance = True
 
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    model = create_model(opt)
    
    ssim_sum = 0
    psnr_sum = 0
    mse_sum = 0
    scores = np.zeros((len(dataset), 3))
    indices = np.array(choose_k_images(len(dataset), 100))
    saved_scores = np.zeros((len(indices), 3))
    
    for i, data in enumerate(dataset):
        generated = model.inference(data['label'], data['inst'], data['image'])
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                               ('synthesized_image', util.tensor2im(generated.data[0])), 
                               ('real_image', util.tensor2im(data['image'][0]))])
        img_path = data['path']
        scores = save_scores(scores, data, generated, i)
        
        if i in indices:
            # save to their visualizerx
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path)
            #save to my files
            save_img(visuals, opt, img_path[0])
            saved_scores[np.where(indices == i)[0]] = scores[i]
            print(i)
        

    webpage.save()
    
    dic_saved = {"viz matrix": saved_scores, "mean": saved_scores.mean(axis = 0)}
    with open(opt.viz_dir + "/saved_results.pkl", 'wb') as f:
        pickle.dump(dic_saved, f)

    mean_scores = scores.mean(axis = 0)
    dic = {"scores matrix": scores, "mean": mean_scores}
    with open(opt.results_dir + "results.pkl", 'wb') as f:
        pickle.dump(dic, f)
        
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    return mean_scores


if __name__ == "__main__": 
    opt = TestOptions().parse()
    print(test(opt))