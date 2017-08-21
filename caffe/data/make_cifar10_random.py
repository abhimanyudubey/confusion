#!/usr/bin/env python
import argparse
import os
import random

parser = argparse.ArgumentParser('Random labels CIFAR10 generation')
parser.add_argument('-i', '--input', help='Path to input folder with cifar10 train.txt/test.txt files', required=True, type=str)
parser.add_argument('-o', '--output', help='Path to output folder (default same as input)', required=False, type=str, default=None)

args = parser.parse_args()

if not args.output:
    args.output = args.input

for txtf in ['train', 'test']:
    outf = open(os.path.join(args.output, '%s_random.txt' % txtf), 'w')
    with open(os.path.join(args.input, '%s.txt' % txtf), 'r') as inpf:
        fnames = []
        labels = []
        for line in inpf:
            fname = line.strip().split()[0]
            label = int(line.strip().split()[1])
            labels.append(label)
            fnames.append(fname)
        random.shuffle(labels)

        for fx, lx in zip(fnames, labels):
            outf.write('%s %d\n' % (fx, lx))
