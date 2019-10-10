import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import cv2
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from PIL import Image
import numpy as np
from html4vision import Col, imagetable
from skimage.measure import compare_ssim, compare_psnr, compare_mse
import pickle
import pdb
from copy import deepcopy
from data_generation import ModelMode

def set_weights(model, weights): 
    """
    Set `weights` as the model weights.
    Tested to work correctly by this:
    assert( all([(model.layers[j].get_weights()[i] == updated_weights[j][i]).all() for j in range(len(updated_weights)) for i in range(len(updated_weights[j]))]) )
    """
    for i, layer in enumerate(model.layers):
        layer.set_weights(weights[i])
    return model

def extract_weights(model): 
    weights = []
    for layer in model.layers:
        weights += [layer.get_weights()]
    return weights

def compute_reptile(new_weights, old_weights, epsilon):
     return [[old_weights[i][j] + (epsilon * (new_weights[i][j] - old_weights[i][j])) for j in range(len(new_weights[i]))] for i in range(len(new_weights))]

def print_viz(data, model): 
    x, y = next(data)
    arr_loss = model.predict_on_batch(x)
    generated = (arr_loss[0] + 1) * 128
    cv2.imwrite("/home/jl5/tmp.png", generated)
    pdb.set_trace()
    return

def train_T_iter(model, T, data): 
    for step in range(T):
        x, y = next(data)
        train_loss = model.train_on_batch(x, y)
        util.printProgress(step, 0, train_loss)
    return model



def reptile_outer_loop(model_name, gpu_id, dbg=False, k=5, T=20):
    network_dir = f'/home/jl5/data/data-posewarp/models/{model_name}' 
    os.makedirs(network_dir)
    params = param.get_general_params()

    img_width = params['IMG_WIDTH']
    img_height = params['IMG_HEIGHT']
    # load the original pretrained weights
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    weight_path = '../models/vgg_100000.h5'
    model.load_weights(weight_path)
    model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    for i in range(params['meta_iterations']+1): 
      print(i)
      # select k images
      data = data_generation.create_feed(params, k, ModelMode.metatrain)
      
      old_weights = deepcopy(extract_weights(model))
      # train on batch for T iterations starting from init/old weights
      model = train_T_iter(model, T, data)
      new_weights = extract_weights(model)
      updated_weights = compute_reptile(new_weights, old_weights, params['epsilon'])
      model = set_weights(model, updated_weights)

      # test every like 300 iterations? 
      if i % params['metamodel_save_interval'] == 0: 
        model.save(network_dir + '/' + str(i) + '.h5')

    return model


if __name__ == "__main__":
    import pdb
    if len(sys.argv) == 3:
        reptile_outer_loop(sys.argv[1], '0,1,2,3')
    elif len(sys.argv) == 4:
        reptile_outer_loop(sys.argv[1], '0,1,2,3', dbg=False, k=int(sys.argv[2]), T=int(sys.argv[3]))
    else:
      print("Wrong num of arguments")
