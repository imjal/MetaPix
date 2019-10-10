import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from tqdm import tqdm
from keras.callbacks import TensorBoard
from time import time
from data_generation import ModelMode


def train(model_name, gpu_id, start_iter=0):
    params = param.get_general_params()

    network_dir = params['model_save_dir'] + '/' + model_name

    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)

    tf_writer = tf.summary.FileWriter(network_dir + "/log/")

    train_feed = data_generation.create_feed(params, None, ModelMode.train)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    model.load_weights('../models/vgg_100000.h5')
    model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    #model.summary()
    n_iters = params['n_training_iter']

    for step in range(start_iter, n_iters+1):
        x, y = next(train_feed)

        train_loss = model.train_on_batch(x, y)

        util.printProgress(step, 0, train_loss)
        summary =tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss)])
        tf_writer.add_summary(summary, step)

        if step > 0 and step % 100 == 0:
          tf_writer.flush()

        if step > 0 and step % params['model_save_interval'] == 0:
            model.save(network_dir + '/' + str(step) + '.h5')
    model.save(network_dir + '/' + str(step) + '.h5')


if __name__ == "__main__":
    if len(sys.argv) == 2:
      train(sys.argv[1], '0,1,2,3')
    else:
      print("Need model name")
