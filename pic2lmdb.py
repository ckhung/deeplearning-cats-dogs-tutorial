#!/usr/bin/python
'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid/Chao-Kuei Hung
Date Created    :20160619
Date Modified   :20170911
usage           :./pic2lmdb.py
python_version  :2.7.*
'''

import argparse, re, subprocess, glob, random, cv2, warnings, lmdb
import numpy as np

import caffe
from caffe.proto import caffe_pb2

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

pcid2nl = {}    # picture class id to numerical label
nl2tl = []      # numerical label to text label

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring()
    )

def pic2lmdb(pics, keep, lmdbpath):
    global pcid2nl, nl2tl
    print 'Creating {}'.format(lmdbpath)
    subprocess.call(['rm', '-f'] + glob.glob(lmdbpath + '/*.mdb'))
    subprocess.call(['mkdir', '-p', lmdbpath])
    included = [0 for x in nl2tl]
    seen = [0 for x in nl2tl]
    n_digits = len(str(len(nl2tl)-1))
    idx_file = open(lmdbpath+'/index.txt', 'w')
    out_lmdb = lmdb.open(lmdbpath, map_size=int(1e12))
    with out_lmdb.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(pics):
            for pcid in pcid2nl:
                m = re.search(r'\b('+pcid+r')\b', img_path)
                if m:
                    break
            if m:
                nl = pcid2nl[pcid]
                tl = nl2tl[nl]
            else:
                warnings.warn('file path "{}" does not contain any pcid, ignored'.format(img_path))
            seen[nl] += 1
            if not keep(seen[nl]):
                continue
            included[nl] += 1
            idx_file.write(img_path+'\n')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, nl)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
    # out_lmdb.close()
    idx_file.close()
    total = 0
    for nl, tl in enumerate(nl2tl):
        total += included[nl]
        print ('{:>5} [{:0>'+str(n_digits)+'}] {}').format(included[nl], nl, nl2tl[nl])
    print '-'*20
    print '{:>5} {}'.format(total, 'TOTAL')
    print

parser = argparse.ArgumentParser(
    description='given a directory of jpg images, create 2 lmdb directories, one for training, the other for validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--group', type=int,
    default=6, help='number of groups in training/validation split')
parser.add_argument('-i', '--index', type=int,
    default=0, help='which group as validation in training/validation split')
parser.add_argument('-s', '--seed', type=int,
    default=0, help='random seed')
parser.add_argument('--training', type=str,
    default='training', help='path for training lmdb')
parser.add_argument('--validation', type=str,
    default='validation', help='path for validation lmdb')
parser.add_argument('LABEL', help='label file')
parser.add_argument('PICS', help='path of pictures')
args = parser.parse_args()

with open(args.LABEL) as f:
    all_labels = f.readlines()

for line in all_labels:
    m = re.search(r'^\s*(\w+)(\s+(\S+))?', line)
    if m:
        tl = m.group(3)
        if not tl in nl2tl:
            nl2tl.append(tl)
        nl = nl2tl.index(tl)    # numerical label
        pcid2nl[m.group(1)] = nl
train_data = sorted([img for img in glob.glob(args.PICS + '/*jpg')])
random.seed(args.seed)
random.shuffle(train_data)
pic2lmdb(train_data, lambda x: x%args.group!=args.index, args.training)
pic2lmdb(train_data, lambda x: x%args.group==args.index, args.validation)

