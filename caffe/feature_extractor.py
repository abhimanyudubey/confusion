#!/usr/bin/env python
# Feature extractor for Caffe models
# Abhimanyu Dubey, 2016
import os
import sys
import argparse
import string
import random
import datetime
import h5py
import lmdb
import json
from collections import OrderedDict
import numpy as np
import threading
import time
import tqdm

''' Opening declarations '''
IGNORE_STRING_FIELDS = ['ceil_mode','sum_pool','norm_region','phase' ,'backend', 'pool', 'operation', 'shuffle', 'lr_mult', 'param', 'mirror', 'coeff', 'bias_term', 'use_global_stats', 'global_pooling', 'variance_norm']
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

def string_arr(inp_str):
    if inp_str is None:
        return None
    try:
        temp_params = json.loads(inp_str)
        if type(temp_params) is list:
            return [str(x) for x in temp_params]
        return temp_params
    except:
        return [str(inp_str)]

def run_command(inp_cmd):
    os.system(inp_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction using h5py with Caffe.')
    parser.add_argument('-m', '--model', type=str, help='Path to model file', required=True)
    parser.add_argument('-w', '--weights', type=str, help='Path to caffemodel with weights', required=True)
    parser.add_argument('-l', '--layers', help='Layers required to be extracted', required=True, type=string_arr)
    parser.add_argument('-o', '--output', help='Output prefix', required=True, type=str)
    parser.add_argument('-i', '--input', help='Custom input file (should be of same type as in model\'s TEST phase, default will be taken from prototxt TEST phase source)', type=str, required=False)
    parser.add_argument('-g', '--gpu', help='GPU ID (default 0)', required=False, default='0')
    parser.add_argument('-b', '--batch_size', help='Custom Batch size to use (override model TEST phase)', required=False)
    parser.add_argument('-a', '--add_headers', help='Write file headers in CSV (default False)', required=False, default=False)
    parser.add_argument('-x', '--add_labels', help='Add labels to output as last column (default False)', required=False, default=False)

    args = parser.parse_args()

    if 'CAFFE_ROOT' not in os.environ:
        print 'CAFFE_ROOT not set, set in environment'
        sys.exit(0)

    os.environ['TEMP_DIR'] = '/tmp/'

    jobID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(25))
    execTime = datetime.datetime.now().strftime("%Y%b%d_%H%M%S")
    tmp_proto = '_'.join([execTime, jobID, 'model.prototxt'])

    net = read_prototxt(args.model)
    # add h5py layer for each input layer

    ind = get_layer(net, 'data')
    if type(ind) is list:
        indn = -1
        for indx in ind:
            if net['layer'][indx]['include']['phase'] == 'TEST':
                indn = indx
                break
        ind = indn

    n_elems = 0
    input_file = ''

    if not args.input:
        if 'image_data_param' in net['layer'][ind]:
            input_file = net['layer'][ind]['image_data_param']['source']
            ft = open(net['layer'][ind]['image_data_param']['source'], 'r').readlines()
            n_elems = len(ft)

        elif 'data_param' in net['layer'][ind]:
            input_file  = net['layer'][ind]['data_param']['source']
            ft = lmdb.open(net['layer'][ind]['data_param']['source'])
            with ft.begin() as txn:
                n_elems = int(txn.stat()['entries'])
            ft.close()
    else:
        input_file = args.input
        if os.path.isdir(input_file):
            ft = lmdb.open(input_file)
            with ft.begin() as txn:
                n_elems = int(txn.stat()['entries'])
            ft.close()
        else:
            ft = open(args.input, 'r').readlines()
            n_elems = len(ft)

        if 'image_data_param' in net['layer'][ind]:
            net['layer'][ind]['image_data_param']['source'] = args.input
        elif 'data_param' in net['layer'][ind]:
            net['layer'][ind]['data_param']['source'] = args.input

    batch_size = 0
    if args.batch_size:
        batch_size = args.batch_size
        args.iter = str(int(n_elems / args.batch_size) + 1)
        if 'image_data_param' in net['layer'][ind]:
            net['layer'][ind]['image_data_param']['batch_size'] = args.batch_size
        elif 'data_param' in net['layer'][ind]:
            net['layer'][ind]['data_param']['batch_size'] = args.batch_size
    else:
        if 'image_data_param' in net['layer'][ind]:
            batch_size = net['layer'][ind]['image_data_param']['batch_size']
        elif 'data_param' in net['layer'][ind]:
            batch_size = net['layer'][ind]['data_param']['batch_size']
        args.iter = str(int(n_elems / batch_size) + 1)

    h5outs = []
    for inp_layer in args.layers:
        if inp_layer in layers(net):
            layer_ind = get_layer(net, inp_layer)
            layer_ind = layer_ind[0]
            # adding layer only if one layer of the name is present
            tmp_h5_file = '_'.join([execTime, jobID, inp_layer.replace('/', '_'), 'output.h5'])
            out_layer = OrderedDict()
            out_layer['name'] = inp_layer + '_h5out'
            out_layer['type'] = 'HDF5Output'
            out_layer['hdf5_output_param'] = OrderedDict()
            out_layer['hdf5_output_param']['file_name'] = os.path.join(os.environ['TEMP_DIR'], tmp_h5_file)
            h5outs.append(os.path.join(os.environ['TEMP_DIR'], tmp_h5_file))
            out_layer['bottom'] = []
            out_layer['bottom'].append(inp_layer)
            out_layer['bottom'].append('label')
            out_layer['include'] = OrderedDict()
            out_layer['include']['phase'] = 'TEST'
            net['layer'].append(out_layer)
        else:
            print 'Incorrect layer', inp_layer, 'input, ignoring'

    # write out new prototxt file
    out_proto_path = os.environ['TEMP_DIR'] + '/' + tmp_proto
    status_file = out_proto_path + '.log'
    status_file_last = out_proto_path + '.log.last'
    write_prototxt(out_proto_path, net)

    output_folder = '/'.join(args.output.split('/')[:-1])
    output_prefix = args.output.split('/')[-1]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    n_iter = int(n_elems/batch_size) + int(n_elems % batch_size > 0)
    # run caffe preds
    print 'Getting predictions from Caffe...'
    caffe_command = '%s/build/tools/caffe test --model=%s --weights=%s --iterations=%d --gpu=%s > %s 2>&1' % (os.environ['CAFFE_ROOT'], out_proto_path, args.weights, n_iter, args.gpu, status_file)
    print 'Caffe command to be run: \n %s' % caffe_command

    feat_thread = threading.Thread(target=run_command, args=(caffe_command,))
    feat_thread.start()

    n_old = 0
    print 'Extracting features:'
    pbar = tqdm.tqdm(total=n_iter)
    while feat_thread.is_alive():
        read_cmd = 'grep "Batch" %s | cut -d " " -f 6,7 | cut -d ","  -f 1 | tail -n 1 > %s' % (status_file, status_file_last)
        os.system(read_cmd)
        ftemp = open(status_file_last, 'r')
        ftempl = ftemp.readlines()
        if len(ftempl) > 0:
            completed_iters = int(ftempl[-1].strip().split()[-1])
        else:
            completed_iters = 0
        pbar.update(completed_iters-n_old)
        n_old = completed_iters
        time.sleep(10.00)
    pbar.close()


    print 'Prediction completed, converting h5 at', os.path.join(os.environ['TEMP_DIR'], tmp_h5_file), 'to output CSV'

    # Converting h5, aggregating results
    feats = []

    for h5file, li in zip(h5outs, args.layers):
        f = h5py.File(h5file, 'r')
        this_feat = []
        nkeys = len(f.keys())
        h5keysgen = ['data'+str(i) for i in range(int(n_iter))]
        h5keyslabel = ['label'+str(i) for i in range(int(n_iter))]
        for h5key, h5keylabel in zip(h5keysgen, h5keyslabel):
            vals = np.array(f[h5key])
            labels = np.array(f[h5keylabel])
            for row_val, label_val in zip(vals,labels):
                out_row_num = row_val.flatten()
                out_label_num = label_val.flatten()
                this_label_elem = float(list(out_label_num)[0])
                if args.add_labels:
                    out_row_num = np.append(out_row_num,this_label_elem)
                this_feat.append(list(out_row_num))
        feats.append(this_feat)

    feats_np = []
    for feat_mat in feats:
        feat_mat_np = np.asarray(feat_mat)[:n_elems]
        feats_np.append(feat_mat_np)

    # writing numpy features to final output CSV
    for feat_np, li in zip(feats_np, args.layers):
        of = open(os.path.join(output_folder, output_prefix+'_'+li.replace('/', '') + '_features.csv'), 'w')
        if args.add_headers:
            if args.add_labels:
                header = ','.join(['feat' + str(i) for i in range(feat_np.shape[1])] + ['label']) + '\n'
            else:
                header = ','.join(['feat' + str(i) for i in range(feat_np.shape[1])]) + '\n'
            of.write(header)

        for out_feat in feat_np:
            of.write(','.join([str(x) for x in out_feat]) + '\n')
        of.close()

    print 'Files written.'
