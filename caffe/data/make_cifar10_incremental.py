#!/usr/bin/env python
import argparse
import os
import random

parser = argparse.ArgumentParser('Incremental labels CIFAR10 generation')
parser.add_argument('-i', '--input', help='Path to input folder with cifar10 train.txt/test.txt files', required=True, type=str)
parser.add_argument('-o', '--output', help='Path to output folder (default same as input)', required=False, type=str, default=None)

args = parser.parse_args()

if not args.output:
    args.output = args.input

fnames = []
labels = []
rand_range = []

with open(os.path.join(args.input, '%s.txt' % 'train'), 'r') as inpf:
       for line in inpf:
           fname = line.strip().split()[0]
           label = int(line.strip().split()[1])
           labels.append(label)
           fnames.append(fname)
       rand_range = range(len(fnames))
       random.shuffle(rand_range)

for i in range(1,10):
    outf = open(os.path.join(args.output, 'train_%f.txt' % (i*0.1)), 'w')
    for j in rand_range[:int(len(rand_range)*0.1*i)]:
        outf.write('%s %d\n' % (fnames[j], labels[j]))
    outf.close()


