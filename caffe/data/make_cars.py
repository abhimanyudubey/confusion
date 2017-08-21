#wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
#wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
##wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
#tar -xvf car_ims.tgz
#wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat

import sys,os
import scipy.io as io

def run_cmd(cmd_str):
  print cmd_str
  os.system(cmd_str)

def get_data():
  run_cmd('wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz')
  run_cmd('tar -xvf cars_train.tgz')
  run_cmd('wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz')
  run_cmd('tar -xvf cars_test.tgz')
  run_cmd('wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz')
  run_cmd('tar -xvf car_devkit.tgz')
  run_cmd('wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat')

def test_data():
  a = io.loadmat('cars_test_annos_withlabels.mat')
  b = a['annotations'][0]
  with open('cars_test.txt', 'w') as fout:
    for t in b:
      outstr = './data/cars_test/%s %d' % (t[5][0],t[4][0][0])
      fout.write(outstr+'\n')
      print outstr

def train_data():
  a = io.loadmat('devkit/cars_train_annos.mat')
  b = a['annotations'][0]
  with open('cars_train.txt', 'w') as fout:
    for t in b:
      outstr = './data/cars_train/%s %d' % (t[5][0],t[4][0][0])
      fout.write(outstr+'\n')
      print outstr


def main():
  get_data()
  test_data()
  train_data()
 
if __name__ == '__main__':
  main()
