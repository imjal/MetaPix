import pickle
import sys
import os

def create_k(txtfile, i):
  f = open(txtfile, "r")
  x = f.readlines()
  list_imgs = [int(img[15:-5])-1 for img in x]

  # create folder + imgs
  root_dir = "/data/jl5/data-meta/dance_vids/"
  orig_dir = os.path.join(root_dir, "test_{i}".format(i=i))
  target_dir = os.path.join(root_dir, "ktest_{i}".format(i=i))
  os.makedirs(target_dir, exist_ok=True)

  for j, img_num in enumerate(list_imgs):
    orig = os.path.join(orig_dir, f'{img_num:08d}.jpg')
    target = os.path.join(target_dir, f'{j+1:08d}.jpg')
    os.system("cp {orig} {target}".format(orig=orig, target=target))


if __name__ == "__main__":
  create_k(sys.argv[1], sys.argv[2])
