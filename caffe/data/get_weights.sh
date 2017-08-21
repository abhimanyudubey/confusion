#!/bin/bash
# Download weights
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -d $DIR/../weights ]; then
	mkdir $DIR/../weights
fi
cd $DIR/../weights

if [ ! -e bvlc_alexnet.caffemodel ]; then
	wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
fi
if [ ! -e bvlc_googlenet.caffemodel ]; then
	wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
fi
if [ ! -e VGG_ILSVRC_16_layers.caffemodel ]; then
	wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
fi

cd -