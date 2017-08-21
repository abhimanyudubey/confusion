#!/usr/bin/env python
import argparse
import glob
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make Caltech101/256 dataset text files')
    parser.add_argument('-i', '--input', help='Path to dataset images', required=True, type=str, default=None)
    parser.add_argument('-o', '--output', help='Output prefix', required=True, type=str, default=None)
    parser.add_argument('-n', '--n_test', help='Number of test points per class', required=False, type=int, default=25)
    args = parser.parse_args()

    class_list = []
    img_classes = glob.glob(os.path.join(args.input, '*'))
    for img_class in img_classes:
        if os.path.isdir(img_class) and img_class.split('/')[-1] not in ['BACKGROUND_Google', '257.clutter']:
            class_list.append(img_class)

    imgs = [glob.glob(os.path.join(x, '*.jpg'))+ glob.glob(os.path.join(x, '*.JPG')) for x in class_list]
    for img_list in imgs:
        random.shuffle(img_list)

    imgs_train = []
    imgs_test = []

    for i, img_list in enumerate(imgs):
        for elem in img_list[:args.n_test]:
            imgs_test.append("%s %d" % (elem.strip(), i))
        for elem in img_list[args.n_test:]:
            imgs_train.append("%s %d" % (elem.strip(), i))

    output_pardir = os.path.dirname(args.output)
    if not os.path.exists(output_pardir):
        os.makedirs(output_pardir)

    random.shuffle(imgs_train)
    random.shuffle(imgs_test)

    with open('%s_train.txt' % args.output, 'w') as of:
        for elem in imgs_train:
            of.write(elem+'\n')

    with open('%s_test.txt' % args.output, 'w') as of:
        for elem in imgs_train:
            of.write(elem + '\n')

    print 'Number of training images: %d, number of validation images: %d' % (len(imgs_train), len(imgs_test))


