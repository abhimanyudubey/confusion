#!/usr/bin/env python
import argparse
import glob
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make CUB dataset text files')
    parser.add_argument('-i', '--input', help='Path to dataset images', required=True, type=str, default=None)
    parser.add_argument('-o', '--output', help='Output prefix', required=True, type=str, default=None)
    parser.add_argument('-l', '--list_path', help='Directory containing list files train.txt and test.txt', required=True, type=str, default=None)
    args = parser.parse_args()

    c_train, c_test = 0, 0
    infile = open(args.list_path+'/train.txt', 'r')
    with open('%s_train.txt' % args.output, 'w') as of:
        for line in infile:
            c_id = int(line.strip().split('.')[0])
            outline = '%s/%s %d\n' % (args.input, line.strip(), c_id)
            of.write(outline)
            c_train+=1

    infile = open(args.list_path + '/test.txt', 'r')
    with open('%s_test.txt' % args.output, 'w') as of:
        for line in infile:
            c_id = int(line.strip().split('.')[0])
            outline = '%s/%s %d\n' % (args.input, line.strip(), c_id)
            of.write(outline)
            c_test+=1

    print 'Number of training images: %d, number of validation images: %d' % (c_train, c_test)


