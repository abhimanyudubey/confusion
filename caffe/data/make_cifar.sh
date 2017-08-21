#!/usr/bin/env bash
# Script to make common data repositories - MNIST, CIFAR10, CIFAR100
# Abhimanyu Dubey, 2016
set -e
if [ -z ${CAFFE_ROOT+x} ]; then
	CAFFE_ROOT=../caffe/build
else
	echo "Using existing CAFFE_ROOT at ${CAFFE_ROOT}"
fi

if [ -z ${DATA_ROOT+x} ]; then
    echo "DATA_ROOT is not set. Exiting."
    exit
fi
# CAFFE_ROOT is set, download imagenet data
if [ ! -d $DATA_ROOT ]; then
    echo "DATA_ROOT does not exist, creating it..."
    mkdir -p $DATA_ROOT
fi

if [ ! -d  $DATA_ROOT/mnist ]; then
    mkdir $DATA_ROOT/mnist
fi

if [ ! -d  $DATA_ROOT/lmdb ]; then
    mkdir $DATA_ROOT/lmdb
fi

OWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DATA_ROOT/mnist
echo "Downloading MNIST..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done

cd $DATA_ROOT/lmdb
# creating mnist lmdb
BACKEND="lmdb"
BUILD=$CAFFE_ROOT/build/examples/mnist

echo "Creating MNIST ${BACKEND}..."

rm -rf mnist_train_${BACKEND}
rm -rf mnist_test_${BACKEND}

$BUILD/convert_mnist_data.bin ../mnist/train-images-idx3-ubyte \
  ../mnist/train-labels-idx1-ubyte mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin ../mnist/t10k-images-idx3-ubyte \
  ../mnist/t10k-labels-idx1-ubyte mnist_test_${BACKEND} --backend=${BACKEND}

echo "MNIST Done"
# MNIST done

if [ ! -d  $DATA_ROOT/cifar10 ]; then
    mkdir $DATA_ROOT/cifar10
fi

cd $DATA_ROOT/cifar10

echo "Downloading CIFAR10..."

wget --no-check-certificate https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo "Unzipping..."

tar -xf cifar-10-python.tar.gz && rm -f cifar-10-python.tar.gz
mv cifar-10-batches-py/* . && rm -rf cifar-10-batches-bin

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

python $OWD/make_cifar10.py

echo "CIFAR10 Done."

if [ ! -d  $DATA_ROOT/cifar100 ]; then
    mkdir $DATA_ROOT/cifar100
fi

cd $DATA_ROOT/cifar100

echo "Downloading CIFAR100..."

wget --no-check-certificate https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

echo "Unzipping..."

tar -xf cifar-100-python.tar.gz && rm -f cifar-100-python.tar.gz
mv cifar-100-python/* . && rm -rf cifar-100-python

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

python $OWD/make_cifar100.py
echo "CIFAR100 Done."

echo "MNIST Train LMDB: $DATA_ROOT/lmdb/mnist_train_lmdb"
echo "MNIST Test LMDB: $DATA_ROOT/lmdb/mnist_test_lmdb"
echo "CIFAR10 Train ImageData: $DATA_ROOT/cifar10/images/train.txt"
echo "CIFAR10 Test ImageData: $DATA_ROOT/cifar10/images/test.txt"
echo "CIFAR10 Train ImageData: $DATA_ROOT/cifar100/images/train.txt"
echo "CIFAR10 Test ImageData: $DATA_ROOT/cifar100/images/test.txt"

