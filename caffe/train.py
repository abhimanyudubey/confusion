#!/usr/bin/env python
''' Function to train neural network in Caffe
    Abhimanyu Dubey, 2016 '''

import os
import sys
from collections import OrderedDict
import argparse
import json
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

''' Opening declarations '''
IGNORE_STRING_FIELDS = ['ceil_mode', 'sum_pool','norm_region','phase' ,'backend', 'pool', 'operation', 'shuffle', 'lr_mult', 'param', 'mirror', 'coeff', 'bias_term', 'use_global_stats', 'global_pooling', 'variance_norm']
IGNORE_STRING_FIELDS_SOLVER = ['test_initialization', 'solver_mode', 'debug_info']
''' Helper functions '''

def get_formatted_input(element):
    try:
        t1 = float(element)
        t2 = int(t1)
    except ValueError:
        return element.replace("\"","")
    else:
        if t1==t2:
            return t2
        else:
            return t1


def get_formatted_dict(inp_dict):
    if type(inp_dict) is list:
        for elem in inp_dict:
            elem = get_formatted_dict(elem)
    elif type(inp_dict) is OrderedDict:
        for key,value in inp_dict.iteritems():
            if type(value) is list:
                for elem in value:
                    elem = get_formatted_dict(elem)
            else:
                inp_dict[key] = get_formatted_dict(value)
    else:
        inp_dict = get_formatted_input(inp_dict)
    return inp_dict


def get_prototxt_format(inp_key,inp_val):
    if type(inp_val) is float:
        return str(inp_val)
    elif type(inp_val) is int:
        return str(inp_val)
    elif type(inp_val) is str:
        if not (type(get_formatted_input(inp_val)) == type(inp_val)):
            return str(get_formatted_input(inp_val))
        if inp_key in IGNORE_STRING_FIELDS:
            return inp_val
        else:
            return '\"'+inp_val+'\"'


def get_prototxt_string(inp_dict,ntabs=0,inp_key=None):
    output_string = ''
    if type(inp_dict) is list:
        for elem in inp_dict:
            output_string += ' '*ntabs + str(inp_key) + get_prototxt_string(elem,ntabs)+'\n'
    elif type(inp_dict) is OrderedDict:
        output_string += ' {\n'
        for key,value in inp_dict.iteritems():
            if type(value) is list:
                output_string += ' '*ntabs + get_prototxt_string(value,ntabs+1,key) + '\n'
            else:
                output_string += ' '*ntabs + str(key) +' ' + get_prototxt_string(value,ntabs+1,key) + '\n'
        output_string += ' '*ntabs + '}\n'
    else:
        output_string += ' : '+get_prototxt_format(inp_key,inp_dict)
    return output_string


def get_trimmed_dictionary(inp_dict):
    if type(inp_dict) is list:
        if len(inp_dict)==1:
            inp_dict = inp_dict[0]
        else:
            for elem in inp_dict:
                elem = get_trimmed_dictionary(elem)
    elif type(inp_dict) is OrderedDict:
        for key,value in inp_dict.iteritems():
            if type(value) is list:
                if len(value) == 1:
                    inp_dict[key] = get_trimmed_dictionary(value[0])
    return inp_dict

''' Main functions '''

def read_prototxt(filename):
    ''' This function reads in a prototxt file (new version) and returns a Python dictionary.
        Usage: read_prototxt(filename) '''
    tokens = []
    with open(filename,'r') as opened_file:
        for line in opened_file:
            trunc_line = line.strip().split('#')[0]
            tokens.append(trunc_line+'\n')
    proto_str = [x for x in ''.join(tokens).replace(":"," : ").replace("\"","").replace(","," , ").replace("{"," { ").replace("}"," } ").replace("\n"," ").split(" ") if not x=='']
    proto_stack = []
    output_dict = OrderedDict()
    if proto_str.count("{")!=proto_str.count("}"):
        return OrderedDict()

    while len(proto_str)>0:
        inp_token = proto_str.pop(0)
        if inp_token == '}':
            temp_dict = OrderedDict()

            stack_entry = proto_stack.pop()
            while not type(stack_entry.values()[0]) is list:
                for key,value in stack_entry.iteritems():
                    if key in temp_dict.keys():
                        temp_dict[key].insert(0,value)
                    else:
                        temp_dict[key]=[value]
                #print proto_stack
                stack_entry = proto_stack.pop()
            if len(stack_entry.values()[0])==0:
                temp_entry = OrderedDict()

                for key,value in stack_entry.iteritems():
                    temp_entry[key] = temp_dict
                proto_stack.append(temp_entry)
        else:
            delimiter = proto_str.pop(0)
            if delimiter == ':':
                # key-value
                temp_dict = OrderedDict()

                temp_dict[inp_token] = proto_str.pop(0)
                proto_stack.append(temp_dict)
            elif delimiter == '{':
                temp_dict = OrderedDict()

                temp_dict[inp_token] = []
                proto_stack.append(temp_dict)
    for elem in proto_stack:
        for k,v in elem.iteritems():
            if k not in output_dict:
                output_dict[k] = [v]
            else:
                output_dict[k].insert(0,v)
    for key,value in output_dict.iteritems():
        output_dict[key] = get_trimmed_dictionary(value)
    final_dict = get_formatted_dict(output_dict)
    final_dict['layer'].reverse()
    return final_dict


def write_prototxt(output_file,prototxt):
    ''' Write prototxt to file.
        Usage: write_prototxt(output_file,prototxt_dictionary) '''
    with open(output_file,'w') as f:
        if 'name' in prototxt.keys():
            f.write('name: \"'+prototxt['name']+'\" \n')
        if 'layer' in prototxt.keys():
            f.write(get_prototxt_string(prototxt['layer'],0,'layer').replace('\n\n','\n'))

def layers(net):
    out = []
    for elem in net['layer']:
        out.append(elem['name'])
        for x in elem['top']:
            out.append(x)
    return set(out)

def get_layer(net,layer_name):
    out = []
    for elem in net['layer']:
        out.append(elem['name'])
    out_ind = []
    for i,elem in enumerate(out):
        if elem == layer_name:
            out_ind.append(i)
    return out_ind

def get_loss_layer(net):
    l_index = -1
    for i,elem in enumerate(net['layer']):
        if elem['top'] == 'loss_train':
            l_index = i
            break
    if l_index >= 0:
        return l_index
    else:
        return None

def add_simplex(network, layerweights, batch_size, normalize=False, label_agnostic = True, is_l1 = True, entropy=False, relu=False):
    # Adding label transformation
    if not label_agnostic and not entropy:
        if 'simplex_label_split' not in network['layer']:
            simplex_labels = OrderedDict()
            simplex_labels['name'] = 'simplex_label_split'
            simplex_labels['type'] = 'Slice'
            simplex_labels['bottom'] = 'label'
            simplex_labels['top'] = ['label_l', 'label_r']
            simplex_labels['slice_param'] = OrderedDict()
            simplex_labels['slice_param']['axis'] = 0
            simplex_labels['slice_param']['slice_point'] = int(0.5*batch_size)
            simplex_labels['include'] = OrderedDict()
            simplex_labels['include']['phase'] = 'TRAIN'
            network['layer'].append(simplex_labels)

            simplex_labeld = OrderedDict()
            simplex_labeld['name'] = 'simplex_label_diff'
            simplex_labeld['type'] = 'Eltwise'
            simplex_labeld['bottom'] = ['label_l', 'label_r']
            simplex_labeld['top'] = 'simplex_label_diff'
            simplex_labeld['eltwise_param'] = OrderedDict()
            simplex_labeld['eltwise_param']['operation'] = 'SUM'
            simplex_labeld['eltwise_param']['coeff'] = [1, -1]
            simplex_labeld['include'] = OrderedDict()
            simplex_labeld['include']['phase'] = 'TRAIN'
            network['layer'].append(simplex_labeld)

            simplex_labela = OrderedDict()
            simplex_labela['name'] = 'simplex_label_abs'
            simplex_labela['type'] = 'AbsVal'
            simplex_labela['bottom'] = 'simplex_label_diff'
            simplex_labela['top'] = 'simplex_label_abs'
            simplex_labela['include'] = OrderedDict()
            simplex_labela['include']['phase'] = 'TRAIN'
            network['layer'].append(simplex_labela)

            simplex_labelt = OrderedDict()
            simplex_labelt['name'] = 'simplex_label_thresh'
            simplex_labelt['type'] = 'Threshold'
            simplex_labelt['bottom'] = 'simplex_label_abs'
            simplex_labelt['top'] = 'simplex_label'
            simplex_labelt['threshold_param'] = OrderedDict()
            simplex_labelt['threshold_param']['threshold'] = 0
            simplex_labelt['include'] = OrderedDict()
            simplex_labelt['include']['phase'] = 'TRAIN'
            network['layer'].append(simplex_labelt)

    for layerx, weightx in layerweights.iteritems():
        if len(get_layer(network, layerx))>0:
            if entropy:
                simplex = OrderedDict()
                simplex['name'] = '%s_flat' % layerx
                simplex['type'] = 'Flatten'
                simplex['bottom'] = layerx
                simplex['top'] = simplex['name']
                simplex['include'] = OrderedDict()
                simplex['include']['phase'] = 'TRAIN'
                network['layer'].append(simplex)

                if relu:
                    simplex = OrderedDict()
                    simplex['name'] = '%s_flat_relu' % layerx
                    simplex['type'] = 'ReLU'
                    simplex['bottom'] = '%s_flat' % layerx
                    simplex['top'] = simplex['name']
                    simplex['include'] = OrderedDict()
                    simplex['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplex)

                    simplexn2 = OrderedDict()
                    simplexn2['name'] = '%s_entropy_prob' % layerx
                    simplexn2['type'] = 'Softmax'
                    simplexn2['bottom'] = '%s_flat_relu' % layerx
                    simplexn2['top'] = simplexn2['name']
                    simplexn2['include'] = OrderedDict()
                    simplexn2['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplexn2)
                else:
                    simplexn2 = OrderedDict()
                    simplexn2['name'] = '%s_entropy_prob' % layerx
                    simplexn2['type'] = 'Softmax'
                    simplexn2['bottom'] = '%s_flat' % layerx
                    simplexn2['top'] = simplexn2['name']
                    simplexn2['include'] = OrderedDict()
                    simplexn2['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplexn2)

                simplexnz = OrderedDict()
                simplexnz['name'] = '%s_entropy_shift' % layerx
                simplexnz['type'] = 'Power'
                simplexnz['bottom'] = '%s_entropy_prob' % layerx
                simplexnz['top'] = simplexnz['name']
                simplexnz['include'] = OrderedDict()
                simplexnz['include']['phase'] = 'TRAIN'
                simplexnz['power_param'] = OrderedDict()
                simplexnz['power_param']['power'] = 1
                simplexnz['power_param']['scale'] = 1
                simplexnz['power_param']['shift'] = 0.00001
                network['layer'].append(simplexnz)

                simplexn2 = OrderedDict()
                simplexn2['name'] = '%s_entropy_log' % layerx
                simplexn2['type'] = 'Log'
                simplexn2['bottom'] = '%s_entropy_shift' % layerx
                simplexn2['top'] = simplexn2['name']
                simplexn2['include'] = OrderedDict()
                simplexn2['include']['phase'] = 'TRAIN'
                network['layer'].append(simplexn2)

                simplex4 = OrderedDict()
                simplex4['name'] = '%s_entropy_eltwise' % layerx
                simplex4['type'] = 'Eltwise'
                simplex4['bottom'] = ['%s_entropy_prob' % layerx, '%s_entropy_log' % layerx]
                simplex4['top'] = simplex4['name']
                simplex4['eltwise_param'] = OrderedDict()
                simplex4['eltwise_param']['operation'] = 'PROD'
                simplex4['include'] = OrderedDict()
                simplex4['include']['phase'] = 'TRAIN'
                network['layer'].append(simplex4)

                simplex5 = OrderedDict()
                simplex5['name'] = '%s_entropy' % layerx
                simplex5['type'] = 'Reduction'
                simplex5['bottom'] = '%s_entropy_eltwise' % layerx
                simplex5['top'] = simplex5['name']
                simplex5['reduction_param'] = OrderedDict()
                simplex5['reduction_param']['operation'] = 'SUM'
                simplex5['reduction_param']['coeff'] = 1.0 / (batch_size)
                simplex5['reduction_param']['axis'] = 0
                simplex5['include'] = OrderedDict()
                simplex5['include']['phase'] = 'TRAIN'
                simplex5['loss_weight'] = weightx
                network['layer'].append(simplex5)

            else:
                simplex = OrderedDict()
                simplex['name'] = '%s_flat_b' % layerx
                simplex['type'] = 'Flatten'
                simplex['bottom'] = layerx
                simplex['top'] = simplex['name']
                simplex['include'] = OrderedDict()
                simplex['include']['phase'] = 'TRAIN'
                network['layer'].append(simplex)

                simplexn15 = OrderedDict()
                simplexn15['name'] = '%s_flat' % layerx
                simplexn15['type'] = 'AbsVal'
                simplexn15['bottom'] = '%s_flat_b' % layerx
                simplexn15['top'] = simplexn15['name']
                simplexn15['include'] = OrderedDict()
                simplexn15['include']['phase'] = 'TRAIN'
                network['layer'].append(simplexn15)

                if normalize:
                    if not is_l1:
                        simplex2 = OrderedDict()
                        simplex2['name'] = '%s_prob' % layerx
                        simplex2['type'] = 'Softmax'
                        simplex2['bottom'] = '%s_flat' % layerx
                        simplex2['top'] = simplex2['name']
                        simplex2['include'] = OrderedDict()
                        simplex2['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplex2)

                        simplex3 = OrderedDict()
                        simplex3['name'] = '%s_split' % layerx
                        simplex3['type'] = 'Slice'
                        simplex3['bottom'] = '%s_prob' % layerx
                        simplex3['top'] = ['%s_l' % simplex3['bottom'], '%s_r' % simplex3['bottom']]
                        simplex3['slice_param'] = OrderedDict()
                        simplex3['slice_param']['axis'] = 0
                        simplex3['slice_param']['slice_point'] = int(0.5*batch_size)
                        simplex3['include'] = OrderedDict()
                        simplex3['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplex3)

                    else:
                        simplexn = OrderedDict()
                        simplexn['name'] = '%s_normalize_power' % layerx
                        simplexn['type'] = 'Power'
                        simplexn['bottom'] = '%s_flat' % layerx
                        simplexn['top'] = simplexn['name']
                        simplexn['include'] = OrderedDict()
                        simplexn['include']['phase'] = 'TRAIN'
                        simplexn['power_param'] = OrderedDict()
                        simplexn['power_param']['power'] = 1
                        simplexn['power_param']['scale'] = 1
                        simplexn['power_param']['shift'] = 1e-4
                        network['layer'].append(simplexn)

                        simplexn15 = OrderedDict()
                        simplexn15['name'] = '%s_normalize_absval' % layerx
                        simplexn15['type'] = 'AbsVal'
                        simplexn15['bottom'] = '%s_normalize_power' % layerx
                        simplexn15['top'] = simplexn15['name']
                        simplexn15['include'] = OrderedDict()
                        simplexn15['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplexn15)

                        simplexn2 = OrderedDict()
                        simplexn2['name'] = '%s_normalize_log' % layerx
                        simplexn2['type'] = 'Log'
                        simplexn2['bottom'] = '%s_normalize_absval' % layerx
                        simplexn2['top'] = simplexn2['name']
                        simplexn2['include'] = OrderedDict()
                        simplexn2['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplexn2)

                        simplex2 = OrderedDict()
                        simplex2['name'] = '%s_prob' % layerx
                        simplex2['type'] = 'Softmax'
                        simplex2['bottom'] = '%s_normalize_log' % layerx
                        simplex2['top'] = simplex2['name']
                        simplex2['include'] = OrderedDict()
                        simplex2['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplex2)

                        simplex3 = OrderedDict()
                        simplex3['name'] = '%s_split' % layerx
                        simplex3['type'] = 'Slice'
                        simplex3['bottom'] = '%s_prob' % layerx
                        simplex3['top'] = ['%s_l' % simplex3['bottom'], '%s_r' % simplex3['bottom']]
                        simplex3['slice_param'] = OrderedDict()
                        simplex3['slice_param']['axis'] = 0
                        simplex3['slice_param']['slice_point'] = int(0.5 * batch_size)
                        simplex3['include'] = OrderedDict()
                        simplex3['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplex3)
                else:
                    simplex3 = OrderedDict()
                    simplex3['name'] = '%s_split' % layerx
                    simplex3['type'] = 'Slice'
                    simplex3['bottom'] = '%s_flat' % layerx
                    simplex3['top'] = ['%s_prob_l' % layerx, '%s_prob_r' % layerx]
                    simplex3['slice_param'] = OrderedDict()
                    simplex3['slice_param']['axis'] = 0
                    simplex3['slice_param']['slice_point'] = int(0.5 * batch_size)
                    simplex3['include'] = OrderedDict()
                    simplex3['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplex3)

                if not label_agnostic:
                    simplex4 = OrderedDict()
                    simplex4['name'] = '%s_diff' % layerx
                    simplex4['type'] = 'Eltwise'
                    simplex4['bottom'] = ['%s_prob_l' % layerx, '%s_prob_r' % layerx]
                    simplex4['top'] = simplex4['name']
                    simplex4['eltwise_param'] = OrderedDict()
                    simplex4['eltwise_param']['operation'] = 'SUM'
                    simplex4['eltwise_param']['coeff'] = [1, -1]
                    simplex4['include'] = OrderedDict()
                    simplex4['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplex4)

                    simplex5 = OrderedDict()
                    simplex5['name'] = '%s_reduction' % layerx
                    simplex5['type'] = 'Reduction'
                    simplex5['bottom'] = '%s_diff' % layerx
                    simplex5['top'] = simplex5['name']
                    simplex5['reduction_param'] = OrderedDict()
                    if is_l1:
                        simplex5['reduction_param']['operation'] = 'ASUM'
                    else:
                        simplex5['reduction_param']['operation'] = 'SUMSQ'
                    simplex5['reduction_param']['coeff'] = 1.0/(batch_size)
                    simplex5['reduction_param']['axis'] = 1
                    simplex5['include'] = OrderedDict()
                    simplex5['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplex5)

                    simplex6 = OrderedDict()
                    simplex6['name'] = '%s_pre_loss' % layerx
                    simplex6['type'] = 'Eltwise'
                    simplex6['bottom'] = ['%s_reduction' % layerx, 'simplex_label']
                    simplex6['top'] = simplex6['name']
                    simplex6['eltwise_param'] = OrderedDict()
                    simplex6['eltwise_param']['operation'] = 'PROD'
                    simplex6['include'] = OrderedDict()
                    simplex6['include']['phase'] = 'TRAIN'
                    network['layer'].append(simplex6)

                    simplex7 = OrderedDict()
                    simplex7['name'] = 'simplex_loss_%s' % layerx
                    simplex7['type'] = 'Reduction'
                    simplex7['bottom'] = '%s_pre_loss' % layerx
                    simplex7['top'] = simplex7['name']
                    simplex7['reduction_param'] = OrderedDict()
                    simplex7['reduction_param']['operation'] = 'SUM'
                    simplex7['reduction_param']['axis'] = 0
                    simplex7['include'] = OrderedDict()
                    simplex7['include']['phase'] = 'TRAIN'
                    simplex7['loss_weight'] = weightx
                    network['layer'].append(simplex7)

                else:
                    if is_l1:
                        simplex4 = OrderedDict()
                        simplex4['name'] = '%s_diff' % layerx
                        simplex4['type'] = 'Eltwise'
                        simplex4['bottom'] = ['%s_prob_l' % layerx, '%s_prob_r' % layerx]
                        simplex4['top'] = simplex4['name']
                        simplex4['eltwise_param'] = OrderedDict()
                        simplex4['eltwise_param']['operation'] = 'SUM'
                        simplex4['eltwise_param']['coeff'] = [1, -1]
                        simplex4['include'] = OrderedDict()
                        simplex4['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplex4)

                        simplex5 = OrderedDict()
                        simplex5['name'] = '%s_reduction' % layerx
                        simplex5['type'] = 'Reduction'
                        simplex5['bottom'] = '%s_diff' % layerx
                        simplex5['top'] = simplex5['name']
                        simplex5['reduction_param'] = OrderedDict()
                        simplex5['reduction_param']['operation'] = 'ASUM'
                        simplex5['reduction_param']['coeff'] = 1.0 / (0.5* batch_size)
                        simplex5['reduction_param']['axis'] = 1
                        simplex5['include'] = OrderedDict()
                        simplex5['include']['phase'] = 'TRAIN'
                        network['layer'].append(simplex5)

                        simplex7 = OrderedDict()
                        simplex7['name'] = 'simplex_loss_%s' % layerx
                        simplex7['type'] = 'Reduction'
                        simplex7['bottom'] = '%s_reduction' % layerx
                        simplex7['top'] = simplex7['name']
                        simplex7['reduction_param'] = OrderedDict()
                        simplex7['reduction_param']['operation'] = 'SUM'
                        simplex7['reduction_param']['axis'] = 0
                        simplex7['include'] = OrderedDict()
                        simplex7['include']['phase'] = 'TRAIN'
                        simplex7['loss_weight'] = weightx
                        network['layer'].append(simplex7)

                    else:
                        simplex7 = OrderedDict()
                        simplex7['name'] = 'simplex_loss_%s' % layerx
                        simplex7['type'] = 'EuclideanLoss'
                        simplex7['bottom'] = ['%s_prob_l' % layerx, '%s_prob_r' % layerx]
                        simplex7['top'] = simplex7['name']
                        simplex7['include'] = OrderedDict()
                        simplex7['include']['phase'] = 'TRAIN'
                        simplex7['loss_weight'] = weightx
                        network['layer'].append(simplex7)

    return network

def get_batch_size(network):
    data_indices = get_layer(network, 'data')
    required_index = -1
    for indx in data_indices:
        layerx = network['layer'][indx]
        try:
            if layerx['include']['phase'] == 'TRAIN':
                required_index = indx
                break
        except:
            required_index = -1
    if 'data_param' in network['layer'][required_index]:
        return network['layer'][required_index]['data_param']['batch_size']
    elif 'image_data_param' in network['layer'][required_index]:
        return network['layer'][required_index]['image_data_param']['batch_size']
    else:
        return None

def get_param_string(param_dict):
    out_str = '_param'
    if not param_dict is None:
        for lx, wx in param_dict.iteritems():
            out_str+= '_%s_%s' % (lx.replace('/','').replace('_',''), str(wx).replace('/','').replace('_',''))
    if param_dict:
        return out_str
    else:
        return ''


def get_simplex_string(param_dict):
    out_str = '_simplex'
    if not param_dict is None:
        for lx, wx in param_dict.iteritems():
            out_str+= '_%s_%s' % (lx.replace('/','').replace('_',''), str(wx).replace('/','').replace('_',''))
    if param_dict:
        return out_str
    else:
        return ''

def train_net(gpu, model_file, solver_file, log_folder, weights_file = None):
    caffe_root = os.environ['CAFFE_ROOT']
    os.environ['GLOG_log_dir'] = log_folder
    caffe_command = '%s/build/tools/caffe train --gpu %s --model=%s --solver=%s' % (caffe_root, gpu, model_file, solver_file)
    if weights_file:
        if weights_file.endswith('.solverstate'):
            caffe_command+= ' --snapshot=%s' % weights_file
        else:
            caffe_command+= ' --weights=%s' % weights_file
    if os.path.isdir(log_folder):
        caffe_command+= ' > %s 2>&1' % os.path.join(log_folder, 'log.txt')
    else:
        caffe_command+= ' > %s 2>&1' % log_folder

    print 'Caffe command to be run: %s\n' % caffe_command
    os.system(caffe_command)

def read_solver(solver_file):
    output_dict = OrderedDict()
    with open(solver_file, 'r') as inp_file:
        for line in inp_file:
            lv = line.strip()
            if not(len(lv) == 0 or lv[0].strip().startswith('#')):
                k,v = [x.strip() for x in lv.split(':')]
                v = get_formatted_input(v)
                if k not in output_dict:
                    output_dict[k] = v
                else:
                    if type(output_dict[k]) is list:
                        output_dict[k].append(v)
                    else:
                        output_dict[k] = [output_dict[k], v]
    return output_dict

def write_solver(output_file, solver):
    with open(output_file, 'w') as out_file:
        for k, v in solver.iteritems():
            is_list = False
            out_v = v
            if type(v) is str and k not in IGNORE_STRING_FIELDS_SOLVER:
                out_v = '"%s"' % v
            elif type(v) is int:
                out_v = '%d' % v
            elif type(v) is float:
                out_v = '%f' % v
            elif type(v) is list:
                for elem in v:
                    out_v2 = elem
                    if type(v) is str and k not in IGNORE_STRING_FIELDS_SOLVER:
                        out_v2 = '"%s"' % elem
                    elif type(v) is int:
                        out_v2 = '%d' % elem
                    elif type(v) is float:
                        out_v2 = '%f' % elem
                    out_file.write('%s : %s\n' % (k, out_v2))
            if not is_list:
                out_file.write('%s : %s\n' % (k, out_v))

def string_dict(inp_str):
    if inp_str is None:
        return None
    temp_params = json.loads(inp_str)
    out = {}
    for k,v in temp_params.iteritems():
        out[str(k)] = v
    return out

def process_and_write_stats(log_prefix, tmp_prefix):

    train_accuracy_command = 'grep "accuracy_train = " %s | cut -d "=" -f 2 | cut -d " " -f 2  > %s_acc_train' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(train_accuracy_command)
    test_accuracy_command = 'grep "accuracy_test = " %s | cut -d "=" -f 2 | cut -d " " -f 2 > %s_acc_test' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(test_accuracy_command)
    train_top5_command = 'grep "top5_train = " %s | cut -d "=" -f 2 | cut -d " " -f 2 > %s_top5_train' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(train_top5_command)
    test_top5_command = 'grep "top5_test = " %s | cut -d "=" -f 2 | cut -d " " -f 2 > %s_top5_test' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(test_top5_command)
    train_iterations_command = 'grep "Iteration " %s | grep "lr" | grep "lr" | cut -d "]" -f 2 | cut -d "," -f 1 | cut -d " " -f 3 > %s_iter_train' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(train_iterations_command)
    test_iterations_command = 'grep "Testing net " %s | cut -d "]" -f 2 | cut -d "," -f 1 | cut -d " " -f 3 > %s_iter_test' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(test_iterations_command)
    train_loss_command = 'grep "loss_train = " %s | cut -d "=" -f 2 | cut -d " " -f 2 > %s_loss_train' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(train_loss_command)
    test_loss_command = 'grep "loss_test = " %s | cut -d "=" -f 2 | cut -d " " -f 2 > %s_loss_test' % (os.path.join(log_prefix, 'log.txt'), tmp_prefix)
    os.system(test_loss_command)

    f = open('%s_acc_train' % tmp_prefix, 'r')
    try:
        trn_acc = [float(x.strip()) for x in f.readlines()]
    except:
        trn_acc = []
    f = open('%s_acc_test' % tmp_prefix, 'r')
    try:
        tst_acc = [float(x.strip()) for x in f.readlines()]
    except:
        tst_acc = []
    f = open('%s_top5_train' % tmp_prefix, 'r')
    try:
        trn_top5 = [float(x.strip()) for x in f.readlines()]
    except:
        trn_top5 = []
    f = open('%s_top5_test' % tmp_prefix, 'r')
    try:
        tst_top5 = [float(x.strip()) for x in f.readlines()]
    except:
        tst_top5 = []
    f = open('%s_iter_train' % tmp_prefix, 'r')
    try:
        trn_iter = [int(x.strip()) for x in f.readlines()]
    except:
        trn_iter = []
    f = open('%s_iter_test' % tmp_prefix, 'r')
    try:
        tst_iter = [int(x.strip()) for x in f.readlines()]
    except:
        tst_iter = []
    f = open('%s_loss_train' % tmp_prefix, 'r')
    try:
        trn_loss = [float(x.strip()) for x in f.readlines()]
    except:
        trn_loss = []
    f = open('%s_loss_test' % tmp_prefix, 'r')
    try:
        tst_loss = [float(x.strip()) for x in f.readlines()]
    except:
        tst_loss = []

    if len(trn_acc)>0:
        best_trn = max(trn_acc)
    else:
        best_trn = 0
        trn_acc = [0]
    if len(tst_acc)>0:
        best_test = max(tst_acc)
    else:
        best_test = 0
        tst_acc = [0]
    if len(trn_top5)>0:
        best_trn5 = max(trn_top5)
    else:
        best_trn5 = 0
        trn_top5 = [0]
    if len(tst_top5)>0:
        best_test5 = max(tst_top5)
    else:
        best_test5 = 0
        tst_top5 = [0]
    if len(trn_loss)>0:
        min_trn = min(trn_loss)
    else:
        min_trn = 0
        trn_loss = [0]
    if len(tst_loss)>0:
        min_test = min(tst_loss)
    else:
        min_test = 0
        tst_loss = [0]
    if len(trn_iter)==0:
        trn_iter = [0]
    if len(tst_iter)==0:
        tst_iter = [0]

    with open(os.path.join(log_prefix, 'train_perf.csv'), 'w') as out_file:
        out_file.write('iteration, accuracy, top5, loss\n')
        for iterx, accx, top5x, lossx in zip(trn_iter, trn_acc, trn_top5, trn_loss):
            out_file.write('%d, %f, %f, %f\n' % (iterx, accx, top5x, lossx))

    with open(os.path.join(log_prefix, 'test_perf.csv'), 'w') as out_file:
        out_file.write('iteration, accuracy, top5, loss\n')
        for iterx, accx, top5x, lossx in zip(tst_iter, tst_acc, tst_top5, tst_loss):
            out_file.write('%d, %f, %f, %f\n' % (iterx, accx, top5x, lossx))

    plt.clf()
    plt.subplot(2,1,1)
    min_len = min(len(trn_iter), len(trn_acc))
    plt.plot(trn_iter[:min_len], trn_acc[:min_len], 'b', label='Training Top-1')
    min_len = min(len(tst_iter), len(tst_acc))
    plt.plot(tst_iter[:min_len], tst_acc[:min_len], 'g', label='Validation Top-1')
    min_len = min(len(trn_iter), len(trn_top5))
    plt.plot(trn_iter[:min_len], trn_top5[:min_len], 'b--', label='Training Top-5')
    min_len = min(len(tst_iter), len(tst_top5))
    plt.plot(tst_iter[:min_len], tst_top5[:min_len], 'g--', label='Validation Top-5')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='upper left')
    plt.subplot(2, 2, 1)
    min_len = min(len(trn_iter), len(trn_loss))
    plt.plot(trn_iter[:min_len], trn_loss[:min_len], 'b', label='Training')
    min_len = min(len(tst_iter), len(tst_loss))
    plt.plot(tst_iter[:min_len], tst_loss[:min_len], 'g', label='Validation')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(log_prefix, 'plot.pdf'), dpi=600)

    return [trn_acc[-1], best_trn, tst_acc[-1], best_test, trn_loss[-1], min_trn, tst_loss[-1], min_test, trn_iter[-1], tst_iter[-1], trn_top5[-1], best_trn5, tst_top5[-1], best_test5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Neural Networks in Caffe')
    parser.add_argument('-m', '--model', help='Path to input model file (required)', type=str, required=True)
    parser.add_argument('-s', '--solver', help='Path to input solver file (required)', type=str, required=True)
    parser.add_argument('-l', '--log_directory', help='Prefix to log directory', type=str, required=True)
    parser.add_argument('-p', '--params', help='Parameters to override in solver', type=string_dict, required=False, default=None)
    parser.add_argument('-u', '--confusion', help='Layers to apply confusion along with weights', type=string_dict, required=False, default = None)
    parser.add_argument('-w', '--weights', help='Path to caffemodel for fine-tuning', required=False, default=None, type=str)
    parser.add_argument('-v', '--snapshot', help='Snapshot Prefix', required=False, default=None, type=str)
    parser.add_argument('-g', '--gpu', help='GPU ID (default 0)', required=False, default='0', type=str)
    parser.add_argument('-d', '--data', help='Data parameters (default loads from prototxt)', required=False, default=None, type=string_dict)
    parser.add_argument('-t', '--print_time', help='Time interval (in seconds) to print (default 10)', required=False, default=10, type=float)
    parser.add_argument('-n', '--normalize', help='Switch for normalization of simplex', required=False, default=False, type=bool)
    parser.add_argument('-a', '--agnostic', help='Loss is label agnostic', required=False, default=True, type=bool)
    parser.add_argument('-i', '--is_l1', help='Use L1 instead of L2 (default False)', action='store_true')
    parser.add_argument('-e', '--entropic', help='Use entropic formulation', action='store_true')
    parser.add_argument('-r', '--relu', help='Use relu at output', action='store_true')

    args = parser.parse_args()
    args.simplex = args.confusion
    args.entropy = args.entropic

    if 'CAFFE_ROOT' not in os.environ:
        print 'Please set CAFFE_ROOT in environment variables.'
        sys.exit(0)

    solver = read_solver(args.solver)
    model = read_prototxt(args.model)

    if args.params:
        for paramx, valx in args.params.iteritems():
            solver[paramx] = valx

    model_name = ''
    if 'name' in model:
        model_name+=model['name']

    log_prefix = ''
    if os.path.isdir(args.log_directory):
        log_prefix = os.path.join(args.log_directory, ('%s_%s%s%s' % (time.strftime("%Y%m%d_%H%M%S", time.gmtime()), model_name, get_simplex_string(args.simplex), get_param_string(args.params)))[:50])
    else:
        log_prefix = args.log_directory

    if args.snapshot:
        snapshot_prefix = os.path.join(args.snapshot, ('%s_%s%s%s' % (time.strftime("%Y%m%d_%H%M%S", time.gmtime()), model_name, get_simplex_string(args.simplex), get_param_string(args.params)))[:50])
        solver['snapshot_prefix'] = snapshot_prefix
        if not os.path.exists(snapshot_prefix):
            os.makedirs(snapshot_prefix)

    batch_size = get_batch_size(model)

    if args.simplex:
        model = add_simplex(model, args.simplex, batch_size, args.normalize, args.agnostic, args.is_l1, args.entropy, args.relu)

    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)

    if args.data:
        data_indices = get_layer(model, 'data')
        trn_index = -1
        tst_index = -1
        set_train = True
        set_test = True
        for indx in data_indices:
            layerx = model['layer'][indx]
            if set_train and (layerx['type'] == 'Data' or layerx['type'] == 'ImageData'):
                if layerx['include']['phase'] == 'TRAIN':
                    trn_index = indx
                    set_train = False
            if set_test and (layerx['type'] == 'Data' or layerx['type'] == 'ImageData'):
                if layerx['include']['phase'] == 'TEST':
                    tst_index = indx
                    set_test = False

        if 'train' in args.data:
            if os.path.exists(args.data['train']):
                if 'data_param' in model['layer'][trn_index]:
                    model['layer'][trn_index]['data_param']['source'] = str(args.data['train'].replace('\'','').replace('"',''))
                elif 'image_data_param' in model['layer'][trn_index]:
                    model['layer'][trn_index]['image_data_param']['source'] = str(args.data['train'].replace('\'','').replace('"',''))
            else:
                print 'Training data argument cannot be reached, please provide absolute path'
                sys.exit(0)
        if 'test' in args.data:
            if os.path.exists(args.data['test']):
                if 'data_param' in model['layer'][tst_index]:
                    model['layer'][tst_index]['data_param']['source'] = str(args.data['test'].replace('\'','').replace('"',''))
                elif 'image_data_param' in model['layer'][tst_index]:
                    model['layer'][tst_index]['image_data_param']['source'] = str(args.data['test'].replace('\'','').replace('"',''))
            else:
                print 'Testing data argument cannot be reached, please provide absolute path'
                sys.exit(0)
        if 'mean_file' in args.data:
            if os.path.exists(args.data['mean_file']):
                model['layer'][trn_index]['transform_param']['mean_file'] = str(args.data['mean_file'].replace('\'','').replace('"',''))
                model['layer'][tst_index]['transform_param']['mean_file'] = str(args.data['mean_file'].replace('\'','').replace('"',''))
            else:
                print 'Mean file argument cannot be reached, please provide absolute path'
                sys.exit(0)

    model_file = os.path.join(log_prefix, 'model.prototxt')
    solver_file = os.path.join(log_prefix, 'solver.prototxt')
    solver['net'] = model_file

    write_prototxt(model_file, model)
    write_solver(solver_file, solver)

    train_thread = threading.Thread(target=train_net, args=(args.gpu, model_file, solver_file, log_prefix, args.weights))
    train_thread.start()
    print 'Caffe Training, Network %s :' % model_name
    print 'Log Directory is : %s' % log_prefix
    max_iter = solver['max_iter']

    time.sleep(30.00)
    while train_thread.is_alive():
        tmp_prefix = '/tmp/%s' % os.path.basename(log_prefix)
        stats = process_and_write_stats(log_prefix, tmp_prefix)
        os.system('clear')
        sys.stdout.write('\t\t==============================================================================================================\n')
        sys.stdout.write('\n\n\t\t%s\n' % model_name)
        sys.stdout.write('\t\t==============================================================================================================\n')
        sys.stdout.write('\t\tLog Directory is : %s\n' % log_prefix)
        sys.stdout.write('\t\tSnapshot Directory is : %s\n' % solver['snapshot_prefix'])
        sys.stdout.write('\t\t==============================================================================================================\n')
        sys.stdout.write('\t\tIteration Count:\t\t\t\t\t\t\t%d/%d\n\n' % (stats[8], max_iter))
        sys.stdout.write('\t\tTraining Accuracy:\t%0.5f\t\t\tBest Accuracy:\t\t%0.5f\n' % (stats[0], stats[1]))
        sys.stdout.write('\t\tTraining Loss:\t\t%1.5f\t\t\tBest Loss:\t\t%1.5f\n'% (stats[4], stats[5]))
        sys.stdout.write('\t\tValidation Accuracy:\t%0.5f\t\t\tBest Accuracy:\t\t%0.5f\n' % (stats[2], stats[3]))
        sys.stdout.write('\t\tValidation Loss:\t%1.5f\t\t\tBest Loss:\t\t%1.5f\n' % (stats[6], stats[7]))
        sys.stdout.write('\t\tTraining Top5:\t\t%0.5f\t\t\tBest Top5:\t\t%0.5f\n' % (stats[10], stats[11]))
        sys.stdout.write('\t\tValidation Top5:\t%0.5f\t\t\tBest Top5:\t\t%0.5f\n' % (stats[12], stats[13]))
        sys.stdout.write('\t\t==============================================================================================================\n')
        sys.stdout.flush()
        time.sleep(args.print_time)
    tmp_prefix = '/tmp/%s' % os.path.basename(log_prefix)
    stats = process_and_write_stats(log_prefix, tmp_prefix)
    os.system('clear')
    sys.stdout.write(
        '\t\t==============================================================================================================\n')
    sys.stdout.write('\n\n\t\t%s\n' % model_name)
    sys.stdout.write(
        '\t\t==============================================================================================================\n')
    sys.stdout.write('\t\tLog Directory is : %s\n' % log_prefix)
    sys.stdout.write('\t\tSnapshot Directory is : %s\n' % solver['snapshot_prefix'])
    sys.stdout.write(
        '\t\t==============================================================================================================\n')
    sys.stdout.write('\t\tIteration Count:\t\t\t\t\t\t\t%d/%d\n\n' % (stats[8], max_iter))
    sys.stdout.write('\t\tTraining Accuracy:\t%0.5f\t\t\tBest Accuracy:\t\t%0.5f\n' % (stats[0], stats[1]))
    sys.stdout.write('\t\tTraining Loss:\t\t%1.5f\t\t\tBest Loss:\t\t%1.5f\n' % (stats[4], stats[5]))
    sys.stdout.write('\t\tValidation Accuracy:\t%0.5f\t\t\tBest Accuracy:\t\t%0.5f\n' % (stats[2], stats[3]))
    sys.stdout.write('\t\tValidation Loss:\t%1.5f\t\t\tBest Loss:\t\t%1.5f\n' % (stats[6], stats[7]))
    sys.stdout.write('\t\tTraining Top5:\t\t%0.5f\t\t\tBest Top5:\t\t%0.5f\n' % (stats[10], stats[11]))
    sys.stdout.write('\t\tValidation Top5:\t%0.5f\t\t\tBest Top5:\t\t%0.5f\n' % (stats[12], stats[13]))
    sys.stdout.write(
        '\t\t==============================================================================================================\n')
    sys.stdout.flush()
