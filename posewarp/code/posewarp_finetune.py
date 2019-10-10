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
import cv2
from skimage.measure import compare_ssim, compare_psnr, compare_mse
import pickle
import numpy as np
from data_generation import ModelMode



def finetune(model_name, save_dir, gpu_id, vid_i, k, T, iter_num):
    params = param.get_general_params()
    img_width = params['IMG_WIDTH']
    img_height = params['IMG_HEIGHT']

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    weight_path = str(os.path.join(params['model_save_dir'], os.path.join(f"{model_name}", f'{iter_num}.h5'))) # model name doesn't super work
    model.load_weights(weight_path)
    model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    # train for T iterations
    train_feed = data_generation.create_feed(params, k, ModelMode.finetune, vid_i=vid_i, txtfile=f'../testset_5_v3/train_{vid_i}_img.txt')
    for step in range(T):
        x, y = next(train_feed)
        train_loss = model.train_on_batch(x, y)
        util.printProgress(step, 0, train_loss)
    
    # test on all items
    test_feed, dir_len = data_generation.create_test_feed(params, k, vid_i=vid_i, txtfile=f'../testset_5_v3/test_{vid_i}_img.txt', k_txtfile=f'../testset_5_v3/train_{vid_i}_img.txt')
    scores = np.zeros((dir_len, 3))
    for j in range(dir_len):
        try:
          x, y, scale, pos, img_num, src = next(test_feed)
          arr_loss = model.predict_on_batch(x)
        except cv2.error as e:
          print("OpenCV Error, gonna ignore")
          continue
        i = 0
        generated = (arr_loss[i] + 1) * 128
        gen_resized = data_generation.reverse_center_and_scale_image(generated, img_width, img_height, pos, scale)
        target = (y[i] + 1) * 128
        target_resized = data_generation.reverse_center_and_scale_image(target, img_width, img_height, pos, scale)
        source = (x[0][i] + 1) * 128
        # resized_source = cv2.resize(source, (0, 0), fx=2, fy=2)
        # source_resized = data_generation.reverse_center_and_scale_image(source, img_width, img_height, pos, scale)
        modified_img = data_generation.add_source_to_image(gen_resized, src)
        cv2.imwrite(save_dir + f'/{img_num:08d}.png', gen_resized)
        scores[j][0] = compare_ssim(gen_resized, target_resized, multichannel=True, data_range=256)
        scores[j][1] = compare_psnr(gen_resized, target_resized, data_range=256)
        scores[j][2] = compare_mse(gen_resized, target_resized)

    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    print(mean_scores)
    print(std_scores)
    save_dict = os.path.join(save_dir, f"saved_scores_{vid_i}.pkl")
    pickle.dump( scores, open( save_dict, "wb" ) )


def finetune_all(model_name, exp_name, gpu_id, k=5, T=20, iter_num=1000): 
  """
  model_name: pass in names of folder that holds weights
  exp_name: folder name of where the visualizations will go
  gpu_id: 0,1,2,3
  iter_num: number of iterations finetuned on - specified in weight names usually
  dbg: do one run of each to see if all the videos work correctly
  """
  for i in range(1, 9):
    print(i)
    # make new folder inside viz/i/generated
    new_path = f'/data/jl5/data-meta/experiments/{exp_name}/viz/{i}/generated/'
    try: 
      os.makedirs(new_path, exist_ok=False)
    except FileExistsError:
      print("Files already exist")
      response = input("Would you like to continue anyways? [Y/n]\n")
      if response != "Y": 
        exit()
    finetune(model_name, new_path, gpu_id, i, k, T, iter_num)

if __name__ == "__main__":
    if len(sys.argv) == 4:
      finetune_all(sys.argv[1], sys.argv[2], '0,1,2,3', iter_num=int(sys.argv[3]))
    elif len(sys.argv) == 5: 
      finetune_all(sys.argv[1], sys.argv[2], sys.argv[3], iter_num=int(sys.argv[4]))
    elif len(sys.argv) == 7:
      finetune_all(sys.argv[1], sys.argv[2], sys.argv[3], k=int(sys.argv[4]), T=int(sys.argv[5]), iter_num=sys.argv[6])
    else:
      print("Need model name")
