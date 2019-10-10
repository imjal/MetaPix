import os
import glob
import numpy as np
"""
Creates the test dataset
"""

def choose_k_images(len_list, k):
    offset = 100
    length = len_list - offset
    if k > 1: 
        step = length // k
        indices = np.arange(offset, step*k +1, step)
    else: 
        step = 1
        indices = np.arange(offset, offset+1, step)
    return indices


def create_k_txtfile(file_img, file_label, newfile_img, newfile_label, k):
    img_list = []
    label_list = []
    with open(file_img, 'r') as f:
        img_list = f.read().splitlines()
    with open(file_label, 'r') as f:
        label_list = f.read().splitlines()
    combined = list(zip(img_list, label_list))
    img_list[:], label_list[:] = zip(*combined)
    indices = choose_k_images(len(img_list), k)

    new_img = [img_list[i] for i in indices]
    new_label = [label_list[i] for i in indices]
    with open(newfile_img, 'w') as f:
            f.writelines("%s\n" % img for img in new_img)
    with open(newfile_label, 'w') as f:
            f.writelines("%s\n" % img for img in new_label)
    return newfile_img, newfile_label


def create_k_txtfile_rand(file_img, file_label, newfile_img, newfile_label, k):
    img_list = []
    label_list = []
    with open(file_img, 'r') as f:
        img_list = f.read().splitlines()
    with open(file_label, 'r') as f:
        label_list = f.read().splitlines()
    combined = list(zip(sorted(img_list), sorted(label_list)))
    img_list[:], label_list[:] = zip(*combined)
    
    indices = np.random.choice(len(img_list), k, replace=False)

    new_img = [img_list[i] for i in indices]
    new_label = [label_list[i] for i in indices]
    with open(newfile_img, 'w') as f:
            f.writelines("%s\n" % img for img in new_img)
    with open(newfile_label, 'w') as f:
            f.writelines("%s\n" % img for img in new_label)
    return newfile_img, newfile_label


def create_train_test_split(name, dataroot, txtfile_img, txtfile_label, split, i):
    img_list = []
    label_list = []
    with open(dataroot + txtfile_img, 'r') as f:
        img_list = f.read().splitlines()
    with open(dataroot + txtfile_label, 'r') as f:
        label_list = f.read().splitlines()
    img_list = np.array(img_list)
    label_list = np.array(label_list)
    assert(len(img_list) == len(label_list))

    split_idx = int(split*len(img_list))
    train_img = img_list[:split_idx]
    train_label = label_list[:split_idx]
    test_img = img_list[split_idx: ]
    test_label = label_list[split_idx: ]

    os.makedirs(name, exist_ok=True)
    paths = []
    lis = [('train_' + str(i) + '_img.txt', train_img), ('train_' + str(i) + '_label.txt', train_label), ('test_' + str(i) + '_img.txt', test_img), ('test_' + str(i) + '_label.txt', test_label)]
    for (x, y) in lis:
        pth = os.path.join(name, x)
        with open(pth, 'w') as f:
            f.writelines("%s\n" % img for img in y)
        paths += [pth]
    return paths

def create_dataset(dir_name, dataroot):
    test_img = []
    test_label = []
    train_img = []
    train_label = []
    for i in range(1, 9):
        lis = create_train_test_split(dir_name, dataroot, 'test_' + str(i) + '_img.txt', 'test_' + str(i) + '_label.txt', 0.85, i)
        test_img+= [lis[2]]
        test_label+= [lis[3]]
        train_label += [lis[1]]
        train_img += [lis[0]]
    return train_img, train_label, test_img, test_label

def copy_test(new_dir_name, test_img, test_label, dataset_name):
    k = len(dataset_name)
    for (x, y) in zip(test_img, test_label): 
        os.system("cp " + x + " " + new_dir_name + "/" + x[k+1:])
        os.system("cp " + y + " " + new_dir_name + "/" + y[k+1:])

def create_k_dataset(dataset_name, k, new_dir_name):
    train_img, train_label, test_img, test_label = load_dataset(dataset_name)
    new_train_img = []
    new_train_label = []
    os.makedirs(new_dir_name, exist_ok=True)
    copy_test(new_dir_name, test_img, test_label, dataset_name)
   
    for i in range(1, 9):
        img, label = create_k_txtfile(dataset_name + "/train_" + str(i) + "_img.txt", dataset_name + "/train_" + str(i) + "_label.txt", \
            new_dir_name + "/train_" + str(i) + "_img.txt", new_dir_name + "/train_" + str(i) + "_label.txt", k)
        new_train_label += [label]
        new_train_img += [img]
    os.system("chmod -R -w " + dataset_name)
    return new_train_img, new_train_label, test_img, test_label


def load_dataset(dir_name):
    L = []
    params = [('test_', '_img'), ('test_', '_label'), ('train_', '_img'), ('train_', '_label')]
    for (x, y) in params:
        L += [glob.glob(dir_name + '/' + x + "*" + y + ".txt")]
    return sorted(L[2]), sorted(L[3]), sorted(L[0]), sorted(L[1])

def load_train_dataset(dir_name): 
    L = []
    params = [('train_', '_img'), ('train_', '_label')]
    for (x, y) in params:
        L += [glob.glob(dir_name + '/' + x + "*" + y + ".txt")]
    return sorted(L[0]), sorted(L[1])


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input to create a data directory')
    parser.add_argument("--k", type=int, help="The number of training images in this dataset")
    opt = parser.parse_args()
    if not (os.path.isdir("testset_split_85")):
        print("--creating dataset")
        txtimg_finetune,txtlabel_finetune, txtimg_test, txtlabel_test = create_dataset("testset_split_85", '/data/jl5/data-meta/')
        print("--created dataset")
    if not (os.path.isdir("testset_" + str(opt.k))):
        print("--creating " + str(opt.k) + " dataset")
        txtimg_finetune,txtlabel_finetune, txtimg_test, txtlabel_test = create_k_dataset("testset_split_85", opt.k, "testset_" + str(opt.k))
        print("--created " + str(opt.k) + " dataset")
    else: 
        print("Dataset already created")



