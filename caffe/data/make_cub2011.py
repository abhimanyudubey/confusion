#!/usr/bin/env python
import argparse
import glob
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make CUB dataset text files')
    parser.add_argument('-i', '--input', help='Path to dataset parent directory', required=True, type=str, default=None)
    parser.add_argument('-o', '--output', help='Output prefix', required=True, type=str, default=None)
    args = parser.parse_args()

    c_train, c_test = 0, 0
    file1 = open(os.path.join(args.input, 'images.txt'),'r')
    file2 = open(os.path.join(args.input, 'train_test_split.txt'), 'r')
    ofile1 = open(os.path.join(args.output, '_train.txt'), 'w')
    ofile2 = open(os.path.join(args.output, '_test.txt'), 'w')

    for i, (line1, line2) in enumerate(zip(file1, file2)):
        lvars1 = line1.strip().split()
        lvars2 = line2.strip().split()

        is_test = bool(int(lvars2[-1]))
        elem_class = int(lvars1[1].split('.')[0])
        elem_fname = os.path.join(args.input, 'images', lvars1[1])
        outline = '%s %d\n' % (elem_fname, elem_class)

        if not is_test:
            ofile2.write(outline)
            c_test+=1
        else:
            ofile1.write(outline)
            c_train+=1

    print '%s training files, %s test files' % (c_train, c_test)
