#!/usr/bin/env bash
# Deep Learning Tools for Caffe
# Abhimanyu Dubey, 2016

CURRENT_DIRECTORY=$(pwd)

if [ -z ${CAFFE_ROOT+x} ]; then
    echo "CAFFE_ROOT is not set. Exiting."
    exit
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
if [ ! -d $DATA_ROOT/tars ]; then
    echo "DATA_ROOT/tars does not exist, creating it..."
    mkdir -p $DATA_ROOT/tars
fi

if [ ! -f $DATA_ROOT/tars/ILSVRC2012_img_train.tar ]; then
    wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar -P $DATA_ROOT/tars/
fi
if [ ! -f $DATA_ROOT/tars/ILSVRC2012_img_val.tar ]; then
    wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar -P $DATA_ROOT/tars/
fi

echo "Training and validation data downloaded..."
# Once files are downloaded, unzipping and setting it up
if [ ! -d $DATA_ROOT/images/train ]; then
    mkdir -p $DATA_ROOT/images/train
fi
if [ ! -d $DATA_ROOT/images/val ]; then
    mkdir -p $DATA_ROOT/images/val
fi6

cd $DATA_ROOT/tars/
tar -xf ILSVRC2012_img_train.tar -C $DATA_ROOT/images/train/
tar -xf ILSVRC2012_img_val.tar -C $DATA_ROOT/images/val/

cd $DATA_ROOT/images/train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

echo "Unzip completed..."

cd $CURRENT_DIRECTORY
# Now that files are downloaded and extracted, we will convert to LMDB for imagenet
mkdir -p $DATA_ROOT/lmdb/
wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz

# Now creating the lmdb from existing files
CAFFE_TOOLS_DIR=$CAFFE_ROOT/build/tools
DATA=$DATA_ROOT/images/lmdb/
TOOLS=$CAFFE_TOOLS_DIR

TRAIN_DATA_ROOT=$DATA_ROOT/images/train/
VAL_DATA_ROOT=$DATA_ROOT/images/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train lmdb..."

rm -fr $DATA/train_lmdb $DATA/train_lmdb_2 $DATA/val_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $DATA/train_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $DATA/train_lmdb_2

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $DATA/val_lmdb

echo "Done."