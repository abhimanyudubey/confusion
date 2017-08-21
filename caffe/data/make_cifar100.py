#!/usr/bin/env python
import cPickle
import scipy.misc
import os
import sys

DATA_ROOT = ''
if os.getenv('DATA_ROOT', False):
    DATA_ROOT = os.getenv('DATA_ROOT')
else:
    print 'DATA_ROOT not set'
    sys.exit(0)


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if not os.path.exists(DATA_ROOT+'/cifar100/images/'):
	os.makedirs(DATA_ROOT+'/cifar100/images/')

if __name__=="__main__":
	train_file = DATA_ROOT+'/cifar100/train'
	test_file = DATA_ROOT+'/cifar100/test'
	train_output_file = DATA_ROOT+'/cifar100/images/train.txt'
	test_output_file = DATA_ROOT+'/cifar100/images/test.txt'
	image_output_dir = DATA_ROOT+'/cifar100/images/'
	with open(train_output_file,'w') as output_file:
		training_batch = unpickle(train_file)
		for data_mat,label,filename in zip(training_batch['data'],training_batch['fine_labels'],training_batch['filenames']):
			img_data = data_mat.reshape(3,32,32).swapaxes(0,2).swapaxes(0,1)
			scipy.misc.imsave(image_output_dir+filename,img_data)
			output_file.write(image_output_dir+filename+' '+str(label)+'\n')
	with open(test_output_file,'w') as output_file:
		training_batch = unpickle(test_file)
		for data_mat,label,filename in zip(training_batch['data'],training_batch['fine_labels'],training_batch['filenames']):
			img_data = data_mat.reshape(3,32,32).swapaxes(0,2).swapaxes(0,1)
			scipy.misc.imsave(image_output_dir+filename,img_data)
			output_file.write(image_output_dir+filename+' '+str(label)+'\n')