#!/usr/bin/env python
import os
import sys
import glob
import urllib
import tarfile
import numpy as np
from scipy.io import loadmat

def download_file(url, dest=None):
    if not dest:
        dest = 'flowers/' + url.split('/')[-1]
    run_cmd = ('wget %s -O %s' % (url, dest))
    print run_cmd
    os.system(run_cmd) 
    #urllib.urlretrieve(url, dest)
    
# Download the Oxford102 flowersset into the current directory
if not os.path.exists('flowers'):
    os.mkdir('flowers')

if not os.path.exists('flowers/102flowers.tgz'):
    print("Downloading images...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')
    tarfile.open("flowers/102flowers.tgz").extractall(path='flowers/')
    
    print("Downloading image labels...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')
    
    print("Downloading train/test/valid splits...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')
    
# Read .mat file containing training, testing, and validation sets.
setid = loadmat('flowers/setid.mat')

# The .mat file is 1-indexed, so we subtract one to match Caffe's convention.
idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

# Read .mat file containing image labels.
image_labels = loadmat('flowers/imagelabels.mat')['labels'][0]

# Subtract one to get 0-based labels
image_labels -= 1

files = sorted(glob.glob('flowers/jpg/*.jpg'))
labels = np.array(zip(files, image_labels))

# Get current working directory for making absolute paths to images
cwd = os.path.dirname(os.path.realpath(__file__))

def write_set_file(fout, labels):
    with open(fout, 'w+') as f:
        for label in labels:
            f.write('%s/%s %s\n' % (cwd, label[0], label[1]))

# Images are ordered by species, so shuffle them
np.random.seed(777)
idx_train = idx_train[np.random.permutation(len(idx_train))]
idx_test = idx_test[np.random.permutation(len(idx_test))]
idx_valid = idx_valid[np.random.permutation(len(idx_valid))]

write_set_file('flowers/train.txt', labels[idx_train,:])
write_set_file('flowers/test.txt', labels[idx_test,:])
write_set_file('flowers/valid.txt', labels[idx_valid,:])

