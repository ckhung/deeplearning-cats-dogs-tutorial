#!/usr/bin/python
# modified from https://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import numpy as np
import argparse, sys, caffe, os, re, warnings, ast

def explicit_path(p):
    return re.search(r'^\.{0,2}/', p)

def read_labels(labelfile):
    pcid2nl = {}    # picture class id to numerical label
    nl2tl = []      # numerical label to text label
    with open(labelfile) as f:
        all_labels = f.readlines()
    prev = ''
    for i, line in enumerate(all_labels):
        m = re.search(r'^\s*(\w+)(\s+(.*))?', line)
        if m:
            tl = m.group(3)         # text label
            if tl != prev:
                if tl in nl2tl:
                    warnings.warn("line {} of {}: 2nd occurrence of '{}' ({}) is not adjacent to the 1st occurrence".format(i+1, args.labels, tl, m.group(1)))
		    tl += '-' + m.group(1)
                nl2tl.append(tl)
                prev = tl
            nl = len(nl2tl)-1       # numerical label
            pcid2nl[m.group(1)] = nl
    return (pcid2nl, nl2tl)

parser = argparse.ArgumentParser(
    description='classify images using bvlc_reference_caffenet',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--caffe', type=str,
    default=os.environ['CAFFE_ROOT'], help='root directory of caffe')
parser.add_argument('-f', '--format', type=str,
    default='top5', help='output format: "top5" or "csv"')
parser.add_argument('--labels', type=str,
    default='data/ilsvrc12/synset_words.txt', help='labels file')
parser.add_argument('--mean', type=str,
    default='[103.939, 116.779, 123.68]', help='mean pixel of training data')
    # visit http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    # Find "mean pixel" within the two "information page" links
parser.add_argument('--model', type=str,
    default='models/bvlc_reference_caffenet/deploy.prototxt', help='model def file (deploy)')
parser.add_argument('--weights', type=str,
    default='models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', help='modeel weights file')
parser.add_argument('--gpu', type=int,
    default=-1, help='gpu id (-1 means cpu-only)')
parser.add_argument('image_files', nargs='*')
args = parser.parse_args()

if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
else:
    caffe.set_mode_cpu()

m = re.match(r'^top(\d+)$', args.format)
if m:
    topn = int(float(m.group(1)))
elif args.format == 'csv':
    topn = 1
else:
    sys.exit('unknown format "{}"'.format(args.format))
    
if args.caffe[-1] != '/':
    args.caffe += '/'
if not explicit_path(args.labels):
    args.labels = args.caffe + args.labels
args.mean = np.array(ast.literal_eval(args.mean))
if not (len(args.mean) == 3 and isinstance(args.mean[0], (int, long, float))):
    sys.exit('correct format is: --mean "[103.939, 116.779, 123.68]"')
if not explicit_path(args.model):
    args.model = args.caffe + args.model
if not explicit_path(args.weights):
    args.weights = args.caffe + args.weights
net = caffe.Net(args.model, args.weights, caffe.TEST) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', args.mean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(1, 3, net.blobs['data'].width, net.blobs['data'].height)

images = [caffe.io.load_image(img_f) for img_f in args.image_files]
transformed_images = [transformer.preprocess('data', img) for img in images]

(pcid2nl, nl2tl) = read_labels(args.labels)
for i in range(len(args.image_files)):
    net.blobs['data'].data[i,...] = transformed_images[i]
output = net.forward()
top_guesses = [prob.argsort()[::-1][:topn] for prob in output['prob']]
print
nlfmt = 'L{:0'+str(len(str(len(nl2tl)-1)))+'d}'
for i in range(len(args.image_files)):
    (fn, img, t_img, top, prob) = (args.image_files[i], images[i],
	    transformed_images[i], top_guesses[i], output['prob'][i][top_guesses[i]])
    if args.format == 'csv':
        print ('{:.3f}, '+nlfmt+', {:10}, {}').format(prob[0], top[0], nl2tl[top[0]], fn)
    else:
        print fn
        for i, t in enumerate(top):
            print ('#{} {:.3f} '+nlfmt+' {} ').format(i+1, prob[i], t, nl2tl[t])
        print
