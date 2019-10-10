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


def test(model_name, save_dir, gpu_id, vid_i, iter_num=9999, dbg=False):
    params = param.get_general_params()
    img_width = params['IMG_WIDTH']
    img_height = params['IMG_HEIGHT']

    test_feed, dir_len = data_generation.create_test_feed(params, 5, vid_i=vid_i, txtfile=f'../testset_5_v3/test_{vid_i}_img.txt', k_txtfile=f'../testset_5_v3/train_{vid_i}_img.txt')

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

    model.summary()
    n_iters = params['n_training_iter']
    gen = np.zeros((dir_len, 3, 256, 256))
    scores = np.zeros((dir_len, 3))

    for j in range(1 if dbg else dir_len):
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
        cv2.imwrite(save_dir + f'/{img_num:08d}.png', modified_img)
        gen[j] = np.transpose(generated, (2, 0, 1))
        scores[j][0] = compare_ssim(generated, target, multichannel=True, data_range=256)
        scores[j][1] = compare_psnr(generated, target, data_range=256)
        scores[j][2] = compare_mse(generated, target)

    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    print(mean_scores)
    print(std_scores)
    save_dict = os.path.join(save_dir, f"saved_scores_{vid_i}.pkl")
    pickle.dump( scores, open( save_dict, "wb" ) )


def test_all(model_name, exp_name, gpu_id, iter_num=9999, dbg=False): 
  """
  model_name: pass in names of folder that holds weights
  exp_name: folder name of where the visualizations will go
  gpu_id: 0,1,2,3
  iter_num: number of iterations finetuned on - specified in weight names usually
  dbg: do one run of each to see if all the videos work correctly
  """
  for i in range(2, 9):
    print(i)
    # make new folder inside viz/i/generated
    new_path = f'/data/jl5/data-meta/experiments/{exp_name}/viz/{i}/generated/'
    try: 
      os.makedirs(new_path, exist_ok=False)
    except FileExistsError:
      print("Files already exist")
      response = input("Would you like to continue anyways?\n")
      if response.lower() != "yes": 
        exit()
    test(model_name, new_path, gpu_id, i, iter_num, dbg)

def test_all_scaled(gpu_id, iter_num=6999, dbg=False): 
  """
  model_name: pass in names of folder that holds weights
  exp_name: folder name of where the visualizations will go
  gpu_id: 0,1,2,3
  iter_num: number of iterations finetuned on - specified in weight names usually
  dbg: do one run of each to see if all the videos work correctly
  """
  model_names = [
    '177_FT_AUTH_vid1', 
    '178_FT_AUTH_2', 
    '179_FT_AUTH_vid3', 
    '180_FT_AUTH_vid4', 
    '181_FT_AUTH_vid5', 
    '182_FT_AUTH_vid6', 
    '183_FT_AUTH_vid7',
    '184_FT_AUTH_vid8'
  ]
  for i in range(1, 9):
    print(i)
    # make new folder inside viz/i/generated
    new_path = f'/data/jl5/data-meta/experiments/185_FT_PT_AUTH_inf/viz/{i}/generated/'
    test(model_names[i-1], new_path, gpu_id, i, iter_num, dbg)


if __name__ == "__main__":
    import pdb
    if len(sys.argv) == 3:
        test_all_scaled(sys.argv[1], int(sys.argv[2]))
    if len(sys.argv) == 5:
        test_all(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) == 6:
        test_all(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
      print("Wrong num of arguments")
