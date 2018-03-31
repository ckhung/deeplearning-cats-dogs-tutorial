#!/usr/bin/python
# https://github.com/BVLC/caffe/issues/290#issuecomment-62846228
import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
    print 'Usage: python {} mean.binaryproto mean.npy'.format(sys.argv[0])
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )

out = out.mean(1).mean(1)
def ppf(x): return "%0.2f" % x 
print '[' + ', '.join(map(ppf, out)) + ']'

