#wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
#wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
##wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
#tar -xvf car_ims.tgz
#wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat

import sys,os
import scipy.io as io
from nabirds import nabirds


def save_data(images_ids, paths, labels, fname):
  with open(fname, 'w') as fout:
    for t in images_ids:
      outstr = './data/nabirds/images/%s %d' % (paths[t],int(labels[t]))
      fout.write(outstr+'\n')
      print outstr

def main():
  dataset_path = './nabirds/'
  train_images, test_images = nabirds.load_train_test_split(dataset_path)
  paths = nabirds.load_image_paths(dataset_path)
  labels = nabirds.load_image_labels(dataset_path)
  #get_data()
  save_data(test_images, paths, labels, 'nabirds/test.txt')
  save_data(train_images, paths, labels, 'nabirds/train.txt')
 
if __name__ == '__main__':
  main()
