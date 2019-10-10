import os
import numpy as np
import cv2
import transformations
import scipy.io as sio
import glob
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pdb
import pickle
import glob
import math
from copy import deepcopy
from enum import Enum

class ModelMode(Enum):
    metatrain = 1
    finetune = 2
    test = 3
    train = 4

def show_box_and_keypoints(img_str, box, x, frame_num):
  """
  Debugging function for checking if the pose matches the image.
  """
  im = np.array(Image.open(img_str), dtype=np.uint8)
  fig,ax = plt.subplots(1)
  x = x[:, :, frame_num] - 1.0
  plt.scatter(x[:, 0], x[:, 1])
  ax.imshow(im)
  coord = box[frame_num, :]
  rect = patches.Rectangle((coord[0],coord[1]),coord[2], coord[3],linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.savefig("/home/jl5/overlay.png")
  pdb.set_trace()


def load_keypoints_boxes(vid_path, i):
  """
  Given a video path, with .pkl file included, load the keypoints & boxes
  """
  keypoints = pickle.load(open(vid_path + "keypoints_vid"+str(i)+".pkl", "rb"))
  keypoints[:, 0, :] = keypoints[:, 0, :]*1.40625 + 280
  keypoints[:, 1, :] = keypoints[:, 1, :]*1.40625
  boxes = pickle.load(open(vid_path + "boxes_vid" + str(i) + ".pkl", "rb"))
  boxes = boxes.reshape(-1, 4)
  boxes[:, 0] = boxes[:, 0]*1.40625 + 280
  boxes[:, 1] = boxes[:, 1]*1.40625
  boxes[:, 2] = boxes[:, 2]*1.40625
  boxes[:, 3] = boxes[:, 3]*1.40625
  return keypoints, boxes


# select relevant keypoints/box from video i based on list_imgs
def grab_subset_vid_i(keypoints, boxes, img_path, list_imgs, k, i, mode_str):
  vid_info = []
  k_keypoints = np.zeros((14, 2, k))
  k_boxes = np.zeros((k, 4))
  for j, img_path_num in enumerate(list_imgs):
    k_keypoints[:, :, j] = keypoints[:, :, img_path_num]
    k_boxes[j] = boxes[img_path_num]
  vid_info.append([{"X": k_keypoints, "bbox": k_boxes, "img_num_info": list_imgs}, 
    k_boxes, k_keypoints, os.path.join(img_path, f"{mode_str}_{i}")])
  return vid_info

def read_img_paths(img_txtfile): 
  f = open(img_txtfile, 'r')
  x = f.readlines()
  f.close()
  list_imgs = [int(img[15:-5]) for img in x]
  return list_imgs

def get_vid_info_list(mode, img_path, keypoints_path, imgnums_txtfile, i, k=None):
  """
  General function to create vid_info_list for all meta-learning datasets
  """
  vids = glob.glob(keypoints_path + 'keypoints_*')
  n_vids = len(vids)
  vid_info_list = []
  if mode == ModelMode.train:
    for i in range(1, n_vids+1):
      keypoints, boxes = load_keypoints_boxes(keypoints_path, i)
      vid_info_list.append([{"X": keypoints, "bbox": boxes}, boxes, keypoints, os.path.join(img_path, f"train_{i}")])
    return vid_info_list
  elif mode == ModelMode.metatrain:
    # grab one random video
    i = np.random.randint(n_vids) + 1
    keypoints, boxes = load_keypoints_boxes(keypoints_path, i)
    # select k frames and append the keypoints & image num to recover image later
    list_imgs = np.random.choice(len(boxes), k)
    return grab_subset_vid_i(keypoints, boxes, img_path, list_imgs, k, i, 'train')
  elif mode == ModelMode.finetune:
    keypoints, boxes = load_keypoints_boxes(keypoints_path, i)
    if k == None: # this means that we are doing k = inf case
      list_imgs = read_img_paths(imgnums_txtfile) # grab all
      len_imgs = len(list_imgs)
      return grab_subset_vid_i(keypoints, boxes, img_path, list_imgs, len_imgs, i, 'test')
    else: # we must select some k images from ith video
      list_imgs = read_img_paths(imgnums_txtfile)
      return grab_subset_vid_i(keypoints, boxes, img_path, list_imgs, k, i, 'test')
  elif mode == ModelMode.test:
    keypoints, boxes = load_keypoints_boxes(keypoints_path, i)
    list_imgs = read_img_paths(imgnums_txtfile)
    len_imgs= len(list_imgs)
    return grab_subset_vid_i(keypoints, boxes, img_path, list_imgs, len_imgs, i, 'test')


def get_person_scale(joints):
  upper_body_size = (-joints[0][1] + (joints[8][1] + joints[11][1]) / 2.0)
  rcalf_size = np.sqrt((joints[9][1] - joints[10][1]) ** 2 + (joints[9][0] - joints[10][0]) ** 2)
  lcalf_size = np.sqrt((joints[12][1] - joints[13][1]) ** 2 + (joints[12][0] - joints[13][0]) ** 2)
  calf_size = (lcalf_size + rcalf_size) / 2.0

  size = np.max([2.5 * upper_body_size, 5.0 * calf_size])
  return size / 200.0


def read_frame(vid_name, frame_num, box, x, start_num=0):
  img_name = os.path.join(vid_name, f'{frame_num+1+start_num:08d}' + '.png')
  if not os.path.isfile(img_name):
    img_name = os.path.join(vid_name, f'{frame_num+1+start_num:08d}' + '.jpg')
  img = cv2.imread(img_name)

  joints = x[:, :, frame_num] - 1.0
  box_frame = box[frame_num, :]
  scale = get_person_scale(joints)
  pos = np.zeros(2)
  pos[0] = (box_frame[0] + box_frame[2] / 2.0)
  pos[1] = (box_frame[1] + box_frame[3] / 2.0)

  show_box_and_keypoints(img_name, box, x, frame_num)

  return img, joints, scale, pos


def read_frame_wimgnum(vid_name, frame_num, box, x, img_num):
  img_name = os.path.join(vid_name, f'{img_num:08d}' + '.png')
  if not os.path.isfile(img_name):
    img_name = os.path.join(vid_name, f'{img_num:08d}' + '.jpg')
  img = cv2.imread(img_name)
  joints = x[:, :, frame_num] - 1.0
  box_frame = box[frame_num, :]
  scale = get_person_scale(joints)
  pos = np.zeros(2)
  pos[0] = (box_frame[0] + box_frame[2] / 2.0)
  pos[1] = (box_frame[1] + box_frame[3] / 2.0)

  show_box_and_keypoints(img_name, box, x, frame_num)
  
  return img, joints, scale, pos

def warp_example_generator(vid_info_list, param, do_augment=True, return_pose_vectors=False):
    img_width = param['IMG_WIDTH']
    img_height = param['IMG_HEIGHT']
    pose_dn = param['posemap_downsample']
    sigma_joint = param['sigma_joint']
    n_joints = param['n_joints']
    scale_factor = param['obj_scale_factor']
    batch_size = param['batch_size']
    limbs = param['limbs']
    n_limbs = param['n_limbs']

    while True:
        x_src = np.zeros((batch_size, img_height, img_width, 3))
        x_mask_src = np.zeros((batch_size, img_height, img_width, n_limbs + 1))
        x_pose_src = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_pose_tgt = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_trans = np.zeros((batch_size, 2, 3, n_limbs + 1))
        x_posevec_src = np.zeros((batch_size, n_joints * 2))
        x_posevec_tgt = np.zeros((batch_size, n_joints * 2))
        y = np.zeros((batch_size, img_height, img_width, 3))
        i = 0
        while i < batch_size:

            # 1. choose random video.
            vid = np.random.choice(len(vid_info_list), 1)[0]

            vid_bbox = vid_info_list[vid][1]
            vid_x = vid_info_list[vid][2]
            vid_path = vid_info_list[vid][3]

            # 2. choose pair of frames
            n_frames = vid_x.shape[2]
            frames = np.random.choice(n_frames, 2, replace=False)
            while abs(frames[0] - frames[1]) / (n_frames * 1.0) <= 0.02:
                frames = np.random.choice(n_frames, 2, replace=False)
            # reindex frames if needed
            if 'img_num_info' in vid_info_list[vid][0]:
              img_num0 = vid_info_list[vid][0]['img_num_info'][frames[0]] + 1
              img_num1 = vid_info_list[vid][0]['img_num_info'][frames[1]] + 1
              I0, joints0, scale0, pos0 = read_frame_wimgnum(vid_path, frames[0], vid_bbox, vid_x, img_num0)
              I1, joints1, scale1, pos1 = read_frame_wimgnum(vid_path, frames[1], vid_bbox, vid_x, img_num1)
            else: 
              I0, joints0, scale0, pos0 = read_frame(vid_path, frames[0], vid_bbox, vid_x)
              I1, joints1, scale1, pos1 = read_frame(vid_path, frames[1], vid_bbox, vid_x)
            
            if scale0 > scale1:
                scale = scale_factor / scale0
            else:
                scale = scale_factor / scale1

            pos = (pos0 + pos1) / 2.0

            if I0 is None:
              print("Img is None\n")
              continue;

            if I1 is None:
              print("Img is None\n")
              continue;

            if scale == 0 or scale == float("inf"):
              print("Scale is not appropriate")
              continue;

            if (joints0 <= -1).all() or (joints1 <= -1).all():
              print("all keypoints are not found")
              continue;


            I0, joints0 = center_and_scale_image(I0, img_width, img_height, pos, scale, joints0)
            I1, joints1 = center_and_scale_image(I1, img_width, img_height, pos, scale, joints1)

            I0 = (I0 / 255.0 - 0.5) * 2.0
            I1 = (I1 / 255.0 - 0.5) * 2.0

            if do_augment:
                rflip, rscale, rshift, rdegree, rsat = rand_augmentations(param)
                I0, joints0 = augment(I0, joints0, rflip, rscale, rshift, rdegree, rsat, img_height, img_width)
                I1, joints1 = augment(I1, joints1, rflip, rscale, rshift, rdegree, rsat, img_height, img_width)

            posemap0 = make_joint_heatmaps(img_height, img_width, joints0, sigma_joint, pose_dn)
            posemap1 = make_joint_heatmaps(img_height, img_width, joints1, sigma_joint, pose_dn)

            src_limb_masks = make_limb_masks(limbs, joints0, img_width, img_height)
            src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
            src_masks = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

            x_src[i, :, :, :] = I0
            x_pose_src[i, :, :, :] = posemap0
            x_pose_tgt[i, :, :, :] = posemap1
            x_mask_src[i, :, :, :] = src_masks
            x_trans[i, :, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            x_trans[i, :, :, 1:] = get_limb_transforms(limbs, joints0, joints1)

            x_posevec_src[i, :] = joints0.flatten()
            x_posevec_tgt[i, :] = joints1.flatten()

            y[i, :, :, :] = I1
            i+=1
        out = [x_src, x_pose_src, x_pose_tgt, x_mask_src, x_trans]

        if return_pose_vectors:
            out.append(x_posevec_src)
            out.append(x_posevec_tgt)

        yield (out, y)

def normalize_pose(pose_i):
  pose_tmp = pose_i -pose_i.min(axis = 0)
  pose_tmp /= pose_tmp.max()
  return pose_tmp


def choose_closest_frame(k_keypoints, pose):
  min_idx = 0
  min_loss = float('inf')
  comp_pose = normalize_pose(pose)
  for i in range(5):
    pose_i = k_keypoints[...,i]
    pose_i = normalize_pose(pose_i)
    dist = np.linalg.norm(pose_i-comp_pose, 2)
    if dist <= min_loss:
      min_loss = dist
      min_idx = i
  return min_idx

def test_generator(kvid_info_list, vid_info_list, param, do_augment=True, return_pose_vectors=False):
    img_width = param['IMG_WIDTH']
    img_height = param['IMG_HEIGHT']
    pose_dn = param['posemap_downsample']
    sigma_joint = param['sigma_joint']
    n_joints = param['n_joints']
    scale_factor = param['obj_scale_factor']
    batch_size = param['batch_size_test']
    limbs = param['limbs']
    n_limbs = param['n_limbs']

    vid_bbox = vid_info_list[0][1]
    vid_x = vid_info_list[0][2]
    vid_path = vid_info_list[0][3]

    k_vid_bbox = kvid_info_list[0][1]
    k_vid_x = kvid_info_list[0][2]
    k_vid_path = kvid_info_list[0][3]
    img_len = len(vid_info_list[0][1])
    for j in range(img_len):
        x_src = np.zeros((batch_size, img_height, img_width, 3))
        x_mask_src = np.zeros((batch_size, img_height, img_width, n_limbs + 1))
        x_pose_src = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_pose_tgt = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_trans = np.zeros((batch_size, 2, 3, n_limbs + 1))
        x_posevec_src = np.zeros((batch_size, n_joints * 2))
        x_posevec_tgt = np.zeros((batch_size, n_joints * 2))
        y = np.zeros((batch_size, img_height, img_width, 3))
        # 2. Pick a frame in order
        if 'img_num_info' in vid_info_list[0][0]:
          img_num = vid_info_list[0][0]['img_num_info'][j] + 1
          I1, joints1, scale1, pos1 = read_frame_wimgnum(vid_path, j, vid_bbox, vid_x, img_num)
        else: 
          raise Exception("lol wut")

        # 3. Choose closest frame for the source image
        idx = choose_closest_frame(k_vid_x, vid_x[..., j])
        I0, joints0, scale0, pos0 = read_frame(k_vid_path, idx, k_vid_bbox, k_vid_x)

        I_orig = deepcopy(I0)
        I_src = scale_crop_src(I_orig)
        
        #show_box_and_keypoints(os.path.join(k_vid_path, f"{idx+1:08d}.jpg"), k_vid_bbox, k_vid_x, idx)
        #show_box_and_keypoints(os.path.join(vid_path, f"{j+1+start_idx:08d}.jpg"), vid_bbox, vid_x, j)

        if scale0 > scale1:
            scale = scale_factor / scale0
        else:
            scale = scale_factor / scale1

        pos = (pos0 + pos1) / 2.0

        if I0 is None:
          print("Img is None\n")
          continue;

        if I1 is None:
          print("Img is None\n")
          continue;

        if scale == 0 or scale == float("inf"):
          print("Scale is not appropriate")
          continue;

        if (joints0 <= -1).all() or (joints1 <= -1).all():
          print("all keypoints are not found")
          continue;

        
        I0, joints0 = center_and_scale_image(I0, img_width, img_height, pos, scale, joints0)
        I1, joints1 = center_and_scale_image(I1, img_width, img_height, pos, scale, joints1)

        I0 = (I0 / 255.0 - 0.5) * 2.0
        I1 = (I1 / 255.0 - 0.5) * 2.0

        

        if do_augment:
            rflip, rscale, rshift, rdegree, rsat = rand_augmentations(param)
            I0, joints0 = augment(I0, joints0, rflip, rscale, rshift, rdegree, rsat, img_height, img_width)
            I1, joints1 = augment(I1, joints1, rflip, rscale, rshift, rdegree, rsat, img_height, img_width)

        posemap0 = make_joint_heatmaps(img_height, img_width, joints0, sigma_joint, pose_dn)
        posemap1 = make_joint_heatmaps(img_height, img_width, joints1, sigma_joint, pose_dn)

        src_limb_masks = make_limb_masks(limbs, joints0, img_width, img_height)
        src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
        src_masks = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

        i = 0 # sorry bad code, assuming batch_size = 1
        x_src[i, :, :, :] = I0
        x_pose_src[i, :, :, :] = posemap0
        x_pose_tgt[i, :, :, :] = posemap1
        x_mask_src[i, :, :, :] = src_masks
        x_trans[i, :, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x_trans[i, :, :, 1:] = get_limb_transforms(limbs, joints0, joints1)

        x_posevec_src[i, :] = joints0.flatten()
        x_posevec_tgt[i, :] = joints1.flatten()

        y[i, :, :, :] = I1
        
        out = [x_src, x_pose_src, x_pose_tgt, x_mask_src, x_trans]

        if return_pose_vectors:
            out.append(x_posevec_src)
            out.append(x_posevec_tgt)

        yield (out, y, scale, pos, img_num, I_src)

def create_test_feed(params, k, vid_i = None, txtfile=None, k_txtfile=None, do_augment=False, return_pose_vectors=False):
  vid_info_list = get_vid_info_list(ModelMode.test, params['img_path'], params['keypoints_path_test'], txtfile, vid_i, k)
  k_vid_info_list = get_vid_info_list(ModelMode.finetune, params['img_path'], params['keypoints_path_test'], k_txtfile, vid_i, k)
  feed = test_generator(k_vid_info_list, vid_info_list, params, do_augment, return_pose_vectors)
  return feed, len(vid_info_list[0][0]['img_num_info'])

def create_feed(params, k, mode, vid_i = None, txtfile=None, k_txtfile=None, do_augment=True, return_pose_vectors=False):
  """
  Create a feed given the mode and some parameters
  """
  if mode == ModelMode.metatrain or mode == ModelMode.train:
    in_mode = "train"
  else:
    in_mode = "test"
  vid_info_list = get_vid_info_list(mode, params['img_path'], params[f'keypoints_path_{in_mode}'], txtfile, vid_i, k)
  feed = warp_example_generator(vid_info_list, params, do_augment, return_pose_vectors)
  return feed

def rand_scale(param):
  rnd = np.random.rand()
  return (param['scale_max'] - param['scale_min']) * rnd + param['scale_min']


def rand_rot(param):
  return (np.random.rand() - 0.5) * 2 * param['max_rotate_degree']


def rand_shift(param):
  shift_px = param['max_px_shift']
  x_shift = int(shift_px * (np.random.rand() - 0.5))
  y_shift = int(shift_px * (np.random.rand() - 0.5))
  return x_shift, y_shift


def rand_sat(param):
  min_sat = 1 - param['max_sat_factor']
  max_sat = 1 + param['max_sat_factor']
  return np.random.rand() * (max_sat - min_sat) + min_sat


def rand_augmentations(param):
  rflip = np.random.rand()
  rscale = rand_scale(param)
  rshift = rand_shift(param)
  rdegree = rand_rot(param)
  rsat = rand_sat(param)
  return rflip, rscale, rshift, rdegree, rsat


def augment(I, joints, rflip, rscale, rshift, rdegree, rsat, img_height, img_width):
  I, joints = aug_flip(I, rflip, joints)
  I, joints = aug_scale(I, rscale, joints)
  I, joints = aug_shift(I, img_width, img_height, rshift, joints)
  I, joints = aug_rotate(I, img_width, img_height, rdegree, joints)
  I = aug_saturation(I, rsat)
  return I, joints


def center_and_scale_image(I, img_width, img_height, pos, scale, joints):
  I = cv2.resize(I, (0, 0), fx=scale, fy=scale)
  joints = joints * scale
  x_offset = (img_width - 1.0) / 2.0 - pos[0] * scale
  y_offset = (img_height - 1.0) / 2.0 - pos[1] * scale
  T = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
  I = cv2.warpAffine(I, T, (img_width, img_height))
  joints[:, 0] += x_offset
  joints[:, 1] += y_offset

  return I, joints

def scale_crop_src(I):
  overlap  = (I.shape[1] - I.shape[0])//2
  I = I[:, overlap:I.shape[1]-overlap]
  I = cv2.resize(I, (512, 512))
  return I

def scale_large_reverse_and_scale(I, img_width, img_height, pos, scale): 
  """
  Takes the difference between the center coords and the original overlap adjusted coordinates
  Then, moves and resizes back to 512 x 512
  """
  width = int(1280 * scale)
  height = int(720 * scale)
  overlap = (width - height)//2
  x_offset = ((pos[0] * scale - overlap) - (img_width/2.0) )* 1/scale  # take difference between real coords and the center coords
  y_offset = ((img_height - 1.0) / 2.0 - pos[1] * scale) * 1/scale
  # y_offset = ((pos[0] * scale) - (img_height/2.0) ) * 1/scale
  T = np.float32([[1, 0, x_offset], [0, 1, -y_offset]])
  I = cv2.resize(I, (0, 0), fx=1/scale, fy=1/scale)
  I = cv2.warpAffine(I, T, (720, 720))
  I = cv2.resize(I, (0, 0), fx=512/I.shape[0], fy=512/I.shape[1])
  return I

def scale_small_reverse_and_scale(I, img_width, img_height, pos, scale):
  #cv2.imwrite("/home/jl5/orig.png", I)

  x_offset = (img_width - 1.0) / 2.0 - pos[0] * scale
  y_offset = (img_height - 1.0) / 2.0 - pos[1] * scale
  width = int(1280 * scale)
  height = int(720 * scale)
  overlap = (width - height)//2
  
  T = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
  T_inv = cv2.invertAffineTransform(T)
  I = cv2.warpAffine(I, T_inv, (img_width, img_height))
  # crop at this step.
  long_side = max(height, width-overlap*2)
  cropped = I[:long_side, overlap:overlap +long_side]
  I = cv2.resize(cropped, (0, 0), fx=512/cropped.shape[0], fy=512/cropped.shape[1])
  #cv2.imwrite("/home/jl5/after.png", I)
  #pdb.set_trace()
  return I


def reverse_center_and_scale_image(I, img_width, img_height, pos, scale):
  height = int(720 * scale)
  width = int(1280 * scale)
  if width < img_width: 
    img = scale_small_reverse_and_scale(I, img_width, img_height, pos, scale)
  else: 
    img = scale_large_reverse_and_scale(I, img_width, img_height, pos, scale)
  return img

def add_source_to_image(I, source):
  new_mask = np.tile(np.any(I == [0, 0, 0], axis = -1)[:, :, None], [1, 1, 3])
  np.copyto(I, source, where=new_mask)
  return I

def aug_joint_shift(joints, max_joint_shift):
  joints += (np.random.rand(joints.shape) * 2 - 1) * max_joint_shift
  return joints


def aug_flip(I, rflip, joints):
  if (rflip < 0.5):
      return I, joints

  I = np.fliplr(I)
  joints[:, 0] = I.shape[1] - 1 - joints[:, 0]

  right = [2, 3, 4, 8, 9, 10]
  left = [5, 6, 7, 11, 12, 13]

  for i in range(6):
      tmp = np.copy(joints[right[i], :])
      joints[right[i], :] = np.copy(joints[left[i], :])
      joints[left[i], :] = tmp

  return I, joints


def aug_scale(I, scale_rand, joints):
  I = cv2.resize(I, (0, 0), fx=scale_rand, fy=scale_rand)
  joints = joints * scale_rand
  return I, joints


def aug_rotate(I, img_width, img_height, degree_rand, joints):
  h = I.shape[0]
  w = I.shape[1]

  center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
  R = cv2.getRotationMatrix2D(center, degree_rand, 1)
  I = cv2.warpAffine(I, R, (img_width, img_height))

  for i in range(joints.shape[0]):
      joints[i, :] = rotate_point(joints[i, :], R)

  return I, joints


def rotate_point(p, R):
  x_new = R[0, 0] * p[0] + R[0, 1] * p[1] + R[0, 2]
  y_new = R[1, 0] * p[0] + R[1, 1] * p[1] + R[1, 2]
  return np.array((x_new, y_new))


def aug_shift(I, img_width, img_height, rand_shift, joints):
  x_shift = rand_shift[0]
  y_shift = rand_shift[1]

  T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
  I = cv2.warpAffine(I, T, (img_width, img_height))

  joints[:, 0] += x_shift
  joints[:, 1] += y_shift

  return I, joints


def aug_saturation(I, rsat):
  I *= rsat
  I[I > 1] = 1
  return I


def make_joint_heatmaps(height, width, joints, sigma, pose_dn):
  height = int(height / pose_dn)
  width = int(width / pose_dn)
  n_joints = joints.shape[0]
  var = sigma ** 2
  joints = joints / pose_dn

  H = np.zeros((height, width, n_joints))

  for i in range(n_joints):
      if (joints[i, 0] <= 0 or joints[i, 1] <= 0 or joints[i, 0] >= width - 1 or
              joints[i, 1] >= height - 1):
          continue

      H[:, :, i] = make_gaussian_map(width, height, joints[i, :], var, var, 0.0)

  return H


def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
  xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                        sparse=False, indexing='xy')

  a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
  b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
  c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

  return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                  2 * b * (xv - center[0]) * (yv - center[1]) +
                  c * (yv - center[1]) * (yv - center[1])))


def make_limb_masks(limbs, joints, img_width, img_height):
  n_limbs = len(limbs)
  mask = np.zeros((img_height, img_width, n_limbs))

  # Gaussian sigma perpendicular to the limb axis.
  sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2

  for i in range(n_limbs):
      n_joints_for_limb = len(limbs[i])
      p = np.zeros((n_joints_for_limb, 2))

      for j in range(n_joints_for_limb):
          p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]

      if n_joints_for_limb == 4:
          p_top = np.mean(p[0:2, :], axis=0)
          p_bot = np.mean(p[2:4, :], axis=0)
          p = np.vstack((p_top, p_bot))

      center = np.mean(p, axis=0)

      sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
      theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

      mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
      mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

  return mask


def get_limb_transforms(limbs, joints1, joints2):
  n_limbs = len(limbs)

  Ms = np.zeros((2, 3, n_limbs))

  for i in range(n_limbs):
      n_joints_for_limb = len(limbs[i])
      p1 = np.zeros((n_joints_for_limb, 2))
      p2 = np.zeros((n_joints_for_limb, 2))

      for j in range(n_joints_for_limb):
          p1[j, :] = [joints1[limbs[i][j], 0], joints1[limbs[i][j], 1]]
          p2[j, :] = [joints2[limbs[i][j], 0], joints2[limbs[i][j], 1]]

      tform = transformations.make_similarity(p2, p1, False)
      Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

  return Ms