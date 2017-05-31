#!/usr/local/bin/python

"""
rsmppl.py
usage: rsmppy.pl model train.pkz test.pkz
Wed Jun 19 17:54:40 JST 2013 daichi@ism.ac.jp

"""

import sys
import gzip
import cPickle
import numpy as np
import scipy as sp

def sigmoid(X):
    return (1 + sp.tanh(X/2))/2

if len(sys.argv) == 1:
   print "usage: rsmppl.py model train.pkz test.pkz"
   sys.exit(0)

fh = open(sys.argv[1], 'r')
model = cPickle.load(fh)
w_vh = model['w_vh']
w_v  = model['w_v']
w_h  = model['w_h']
fh.close ()

print 'loading training data..'
fh = gzip.open(sys.argv[2]); train = cPickle.load(fh); fh.close();
print 'loading test data..'
fh = gzip.open(sys.argv[3]); test = cPickle.load(fh); fh.close();

trainD = train.sum(axis=1)
testD  = test.sum(axis=1)

n = train.shape[0]

# compute hidden activations
h = sigmoid(np.dot(train, w_vh) + np.outer(trainD, w_h))
# compute visible activations
v = np.dot(h, w_vh.T) + w_v
# exp and normalize.
tmp = np.exp(v)
sum = tmp.sum(axis=1)
sum = sum.reshape((n,1))
pdf = tmp / sum

z = np.nansum(test * np.log(pdf))
s = np.sum(test)
ppl = np.exp(- z / s)
print "PPL =", ppl


