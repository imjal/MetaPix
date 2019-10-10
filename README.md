# [MetaPix: Few-Shot Video Retargeting](https://imjal.github.io/MetaPix/)

If this code helps with your work/research, please consider citing
Jessica Lee, Deva Ramanan and Rohit Girdhar. MetaPix: Few-Shot Video Retargeting. arXiv preprint arXiv:1910.04742, 2019.

```bibtex
@article{lee2019metapix,
    title={{MetaPix: Few-Shot Video Retargeting}},
    author={Lee, Jessica and Ramanan, Deva and Girdhar, Rohit},
    journal={arXiv preprint arXiv:1910.04742},
    year={2019}
}
```

## Prerequisites

This code has been tested on a Linux (CentOS 7) system, though should be compatible with any OS that can run python, PyTorch, and Keras.

### Anaconda Installation

We have installed the following requirements through anaconda with python version 3.6.8, and we encourage you to use anaconda environments in order to simplify and make things easier.

```bash
$ cd metapix
$ conda env create -f metapix_environment.yml
```

### Other Package Managers

Here's a general laundry list of what you should install:

* PyTorch
* Keras
* Skimage
* OpenCV
* Tensorflow (for Tensorboard)
* SciPy
* Pillow (PIL)
* Pip
* HTML4Vision

## Setting up Data Directory

### Data Directory Structure

The following is the data directory structure that the code expects. We have a parent folder called `data-meta`, and then we have a bunch of subdirectories. For each experiment run we have `experiments` directory, which stores each experiment results in a separate folder (i.e. 001_TRAIN_PT). All experiments are enumerated and described by the mode (training, testing, meta-training, finetuning).

We also have the data folders, separated by the video ID, train versus test, and whether it holds images or labels. The images must be numbered as `img_%08d.png` and labels as `label_%08d.png`.

```
. # some data storage location
├── ...
├── data-meta
│   ├── experiments # store all experiments
│   │     ├── 001_TRAIN_PT
│   │     ├── 002_TEST_001
│   │     ├── 003_TEST_PW # also store results of posewarp
│   │     └── ...
│   ├── train_1_img
│   ├── train_2_label
│   ├── ...
│   ├── train_10_img
│   ├── train_10_label
│   ├── test_1_img
│   ├── test_1_label
│   └── ...
└── data-posewarp
│   ├── models # store posewarp models
│   │     ├── AUTH # store posewarp pretrained model here
│   │     ├── 003_TRAIN_PW
│   │     └── ...
│   ├── train_keypoints
│   │     ├── boxes_vid1.pkl
│   │     ├── keypoints_vid1.pkl
│   │     └── ...
│   └── test_keypoints # same as train format
└──
```

<!-- For ease of reproduction, you can download our data here: [name](link) (`.tgz` x GB).  -->

### Dataset

You can find the youtube id's of the videos we used for our train and test set in `datasets_used.txt`.

## Base Retargeting Architecture: Pix2PixHD

For more extensive ablation experiments, we stricly focus on the Pix2Pix architecture. All pathnames referenced in this section assume you have cd'ed into the directory `pix2pixHD`.

### Configure Data Directory with Pix2PixHD Directory

First, edit the file `pix2pixHD/options/base_options.py` such that the option `--dataroot` points to your directory. All of our scripts use the default dataroot.

Then, importantly, you must symlink the directory called data-meta in the root of `metapix/pix2pixHD/data-meta/` to this location.

```bash
$ cd pix2pixHD
$ ln -s data-meta /path/to/data_dir
```

### Running Experiments

Some sample experiments are included in the directory `experiments`. These are bash scripts of python commands with option configurations that run the `launch.py` script. Note that all runs have to be called from the root pix2pixHD directory.

```bash
$ cd pix2pixHD
$ experiments/071_FT_PT_ALL.sh # k = inf, T = inf baseline
```

We have the following modes and their description:

* _train_: Train a normal Pix2PixHD model.
* _meta-train_: train a new initialization using Reptile.
* _finetune_: Given initial weights, finetune in a k, T setting and test performance after finetuning.
* _test_: Given initial weights for each test video, get test performance.
* _test_checkpoints_: Will run a script that will test for validation performance each new saved checkpoint for another training run.
* _test_all_list_: Given a list of model checkpoints, test each of the weights.
* _test_theta_init_: Given an existing checkpoints directory with intial weights, run the intitialization as if it's a model itself.

### Testing

In order to compare fairly across all methods, a testing class called `MetaPixTest` will  compares each output of each method (Pix2PixHD versus Posewarp) to the ground truth image, and it also has additional functionality such as computing the temporal coherence metric (standard deviation per pixel), and visualize a subset of the outputs. You can view it in file `visualizations/golden_testset.py`. We load all the experiment directories, that must have `data-meta/experiments/{exp_name}/viz/{vidnum}/generated/` folder for each test video that contains a generated image for all of the labels in the `test_{vidnum}_label.txt` file.

```python
from golden_testset import MetaPixTest, FileType
str_base = '/home/jl5/data/data-meta/experiments/'
dirs = {
    "name of experiment": "/home/jl5/..." # experiment directory path
    ...
}
m = MetaPixTest(dirs, FileType.png, data_root, path_to_store_scores)
m.create_test_set()
m.generate_visualization()
mean_scores = m.test_images()
```

You can find sample runs of the visualizations that were used in order to generate visualizations of the videos in `pix2pixHD/visualizations/`.

You can specify the experiment name and the experiment directory in order for it to be tested against the ground truth. In order to visualize the results of a subset of results, you can call the generate_visualization function that will select a subset of 100 results that will generate an HTML file in the new folder that lists all the images side-by-side. Warning: it was configured to work on my machine, so you may have to look at [HTML4Vision](https://github.com/mtli/HTML4Vision) (created by a labmate!) pathrep documentation to make it work for you.

If you wish to test another method, you should make sure that the images that your method output are 1) 512 x 512, 2) all have same file format, 3) produce an output regardless of the method's internal algorithm of cropping that is aligned with the ground truth image, which is just a center crop of the original 1280x720 image.

## Base Retargeting Architecture: Posewarp

This code only updates the generator network. To update multiple models (like a generator and discriminator in the case of GANs), extract weights from multiple models and perform the same update operations. For more details, file `reptile_implementation.py` contains the further implementation details.

### Configuring the Repository

Open `posewarp/code/param.py` and edit the following fields with your file paths: project_dir, model_save_dir, data_dir, and save_img_dir. We actually use different data repositories for this method, because this method requires cropping and scaling, meaning we need the full size original image. We also store the keypoints and boxes since this method requires that, which we provide in the [link](dummy). You can point the directory which stores the pickle files that hold the keypoints per video in the fields keypoints_path_train and keypoints_path_test.

Move the author weights `vgg_1000000.h5` into the folder `data-posewarp/models/AUTH/`, and rename it to `1000000.h5`. This will allow you to run the sample run below.

### Sample Runs

Sample runs can be found at `posewarp/code/exp/`. These also need to be called from the `posewarp` directory.
