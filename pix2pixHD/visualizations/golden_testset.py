from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
import os
from skimage.measure import compare_ssim, compare_psnr, compare_mse
import numpy as np
from html4vision import Col, imagetable
import glob
import pdb
import sys
sys.path.append("../code/")
from data_loader import choose_k_images
import pickle
from statistics import mean
import cv2

class FileType(Enum):
    jpg = 1
    png = 2

class TestObject(ABC):
    def assert_all_images(self, direc, mode):
        # check img sizes are all 512 x 512
        for i in range(1, 9):
            dir_list = glob.glob(direc + f'/viz/{i}/generated/*')
            im = Image.open(dir_list[0])
            width, height = im.size
            assert(width == 512, "Width is not 512")
            assert(height == 512, "Height is not 512")

            # img format
            _, ext = os.path.splitext(dir_list[0])
            ext = ext.lower()[1:] # get rid of front
            assert(ext == mode.name)

    @abstractmethod
    def create_test_set(self):
        """This function will generate the test set as specified by the ground truth directory."""
        pass

    def __init__(self, dirs, mode, root_img_dir, exp_root_dir):
        for direc in dirs.values():
            self.assert_all_images(direc, mode)
        self.dirs = dirs
        self.mode = mode
        self.root_img_dir = root_img_dir
        self.exp_root_dir = exp_root_dir

    def get_directories(self):
        return self.dirs


    @abstractmethod
    def test_images(self):
        # line up images by saved number

        # define all the methods you want to use to test

        # return a dictionary that saves all the stuff + save through pickle
        return

    @abstractmethod
    def generate_visualization(self):
        #line up images number
        # generate the visualization using HTML
        # output html in the experiment directory
        return

class MetaPixTest(TestObject):
    """
    Requires all to be 512x512
    Will only accept images that have the same format and location similar
    """

    def __init__(self, dirs, mode, root_img_dir, exp_root_dir, debug):
        TestObject.__init__(self, dirs, mode, root_img_dir, exp_root_dir)
        self.visualize_set = []
        self.entire_set = []
        self.all_scores = []
        self.debug = debug
        try:
            os.makedirs(exp_root_dir, exist_ok=False)
        except FileExistsError:
            print("Files already exist")
            response = input("Would you like to continue anyways? [Y/n]\n")
            if response != "Y":
                exit()

    def create_test_set(self):
        """
        Sets member variables visualize_set and entire_set
        """
        visualize_set = []
        entire_set = [None] * 8
        test_set_indices = []
        for i in range(8):
            img_list = open(f"/home/jl5/fewshot_pix2pix/datasets/testset_5_v3/test_{i+1}_img.txt").readlines()
            entire_set[i] = len(img_list)
            test_set_indices += [[int(path.strip()[15:-4]) for path in img_list]]
            indices = choose_k_images(entire_set[i], 100)
            visualize_set += [[img_list[i].strip() for i in range(len(img_list)) if i in indices]]
        self.visualize_set = visualize_set
        self.entire_set = entire_set
        self.test_set_indices = test_set_indices


    def test_video(self, gt_dir, vid_dir, vid_i):
        scores_i = np.zeros((self.entire_set[vid_i], 3))
        for i, j in enumerate(self.test_set_indices[vid_i][:1] if self.debug else self.test_set_indices[vid_i]):
            img = np.array(Image.open(os.path.join(vid_dir, f"{j:08d}.{self.mode.name}")))
            gt_img = np.array(Image.open(os.path.join(gt_dir, f"img_{j:08d}.{self.mode.name}")))
            scores_i[i][0] = compare_ssim(img, gt_img, multichannel=True, data_range=256)
            scores_i[i][1] = compare_psnr(img, gt_img, data_range=256)
            scores_i[i][2] = compare_mse(img, gt_img)
        return scores_i


    def test_images(self):
        mean_scores = np.zeros((len(self.dirs), 8, 3))
        all_scores = []
        for vid in range(len(self.dirs)):
            vid_scores = [None] * 8
            for i in range(8):
                # get proper directory
                gt_dir = os.path.join(self.root_img_dir, f"test_{i+1}_img")
                formatted_dir = os.path.join(list(self.dirs.values())[vid], f'viz/{i+1}/generated/')
                # test against each image correctly
                vid_scores[i] = self.test_video(gt_dir, formatted_dir, i)
                mean_scores[vid][i] = vid_scores[i].mean(axis=0)
                print(mean_scores[vid][i])
            all_scores += [vid_scores]
        self.all_scores = all_scores
        tot_mean_scores = mean_scores.mean(axis=1)
        save_dict = {"all_scores": all_scores, "vidbyvid": mean_scores, "mean_scores": tot_mean_scores, "std_scores": np.std(tot_mean_scores, axis = 0)}
        with open(self.exp_root_dir + f"results.pkl", "wb") as f:
            pickle.dump(save_dict, f)
        return tot_mean_scores

    def compute_temporal_coherence(self, debug=False):
        mean_scores = np.zeros((len(self.dirs)))
        all_scores = []
        for vid in range(len(self.dirs)):
            vid_scores = [None] * 8
            for j in range(8):
                # get proper directory
                gt_dir = os.path.join(self.root_img_dir, f"test_{j+1}_img")
                formatted_dir = os.path.join(list(self.dirs.values())[vid], f'viz/{j+1}/generated/')
                # load all images from directory into numpy array
                list_imgs = os.listdir(formatted_dir)
                list_imgs = [ formatted_dir + x  for x in list_imgs]
                A = np.zeros((len(list_imgs), 512, 512, 3))
                for i in range(100 if debug else len(list_imgs)):
                    A[i] = cv2.imread(list_imgs[i])
                # test against each image correctly
                vid_scores[j] = np.nanstd(np.nanmean(A, axis=(1,2,3)))
            mean_scores[vid] = mean(vid_scores)
            all_scores += vid_scores
        save_dict = {"all": all_scores, "mean_scores": mean_scores}
        with open(self.exp_root_dir + f"temporal_results.pkl", "wb") as f:
            pickle.dump(save_dict, f)
        return mean_scores

    def generate_visualization(self, subset = False, num_output_imgs=10):
        # grab all dirs to grab all images
        mode = self.mode.name
        for i in range(8):
            cols = []
            imgs = self.visualize_set[i]
            if subset:
                step = len(imgs) // num_output_imgs
                indices = np.arange(0, step*num_output_imgs+1, step)
                imgs = [imgs[i] for i in range(len(imgs)) if i in indices]
            gt_dir = [os.path.join(self.root_img_dir, im_ext) for im_ext in imgs]
            cols += [Col('img', "GT", gt_dir)]
            for k, v in self.dirs.items():
                viz_set_list = [os.path.join(v, f'viz/{i+1}/generated/{im_ext[15:]}') for im_ext in imgs]
                cols += [Col('img', k, viz_set_list)]
            # output html
            outfile = self.exp_root_dir + f"testset_viz_{i}.html"
            imagetable(cols, out_file=outfile, title=f"Results for Video {i+1}", pathrep=('/home/jl5/data/data-meta/', '../../'))

