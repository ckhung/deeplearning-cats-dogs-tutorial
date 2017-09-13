#!/usr/bin/python
# modified from https://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import numpy as np
import argparse, sys, caffe, os, re

def explicit_path(p):
    return re.search(r'^\.{0,2}/', p)

parser = argparse.ArgumentParser(
    description='classify images using bvlc_reference_caffenet',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--caffe', type=str,
    default=os.environ['CAFFE_ROOT'], help='root directory of caffe')
parser.add_argument('-t', '--top', type=int, default=5, help='top-most how many guesses')
parser.add_argument('-v', '--verbose', type=int, default=1, help='verbosity')
parser.add_argument('-w', '--width', type=int, default=100, help='line width of pretty print')
parser.add_argument('--labels', type=str,
    default='data/ilsvrc12/synset_words.txt', help='labels file')
parser.add_argument('--mean', type=str,
    default='python/caffe/imagenet/ilsvrc_2012_mean.npy', help='mean image npy file')
parser.add_argument('--model', type=str,
    default='models/bvlc_reference_caffenet/deploy.prototxt', help='model def file (deploy)')
parser.add_argument('--weights', type=str,
    default='models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', help='modeel weights file')
parser.add_argument('image_files', nargs='*')
args = parser.parse_args()

caffe.set_mode_cpu()
if args.caffe[-1] != '/':
    args.caffe += '/'
if not explicit_path(args.labels):
    args.labels = args.caffe + args.labels
if not explicit_path(args.mean):
    args.mean = args.caffe + args.mean
if not explicit_path(args.model):
    args.model = args.caffe + args.model
if not explicit_path(args.weights):
    args.weights = args.caffe + args.weights
net = caffe.Net(args.model, args.weights, caffe.TEST) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(args.mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(999, 3, 227, 227)

images = [caffe.io.load_image(img_f) for img_f in args.image_files]
transformed_images = [transformer.preprocess('data', img) for img in images]

with open(args.labels) as f:
    all_lines = f.readlines()
labels = []
for line in all_lines:
    m = re.search(r'^\s*(\w+)(\s+(.*))?', line)
    if m:
        tl = m.group(3)
        pcid = m.group(1)
        name = m.group(3) if m.group(3) else pcid
        labels.append((pcid, name))
# labels = np.array(labels)
for i in range(len(args.image_files)):
    net.blobs['data'].data[i,...] = transformed_images[i]
output = net.forward()
top_guesses = [prob.argsort()[::-1][:args.top] for prob in output['prob']]
print
for i in range(len(args.image_files)):
    (fn, img, t_img, top, prob) = (args.image_files[i], images[i],
	    transformed_images[i], top_guesses[i], output['prob'][i][top_guesses[i]])
    if args.verbose > 0:
        print fn
        if args.verbose > 1:
            print '# {} => {}'.format(img.shape, t_img.shape)
            tmp = [c for x in img for y in x for c in y]
            print '# hist of orig {}'.format(np.histogram(tmp, bins=5))
            tmp = [c for x in t_img for y in x for c in y]
            print '# hist of trans {}'.format(np.histogram(tmp, bins=5))
    for i, t in enumerate(top):
        print '#{} {:.3f} {:10} {:}'.format(i+1, prob[i], labels[t][0], labels[t][1])
    print
