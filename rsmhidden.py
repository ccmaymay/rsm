#!/bin/env python

"""
rsmhidden.py
hidden activation reconstruction.
usage: rsmhidden.py model data
$Id: rsmhidden.py,v 1.1 2013/06/29 02:00:27 daichi Exp $
"""

import sys
import pickle
import fmatrix
import numpy as np
from rsm_numpy import sigmoid

if len(sys.argv) == 1:
    print "usage: % rsmhidden.py model data"
    print "$Id: rsmhidden.py,v 1.1 2013/06/29 02:00:27 daichi Exp $"
    sys.exit(0)
else:
    file = sys.argv[1]
    data = sys.argv[2]

with open(file, 'r') as fh:
    model = pickle.load(fh)
w_vh = model['w_vh']
w_v  = model['w_v']
w_h  = model['w_h']

X = fmatrix.parse(data)
X.resize((X.shape[0],w_vh.shape[0]))

for i in xrange(X.shape[0]):
    d = X[i]
    h = sigmoid(np.dot(d, w_vh) + np.sum(d) * w_h)
    write = sys.stdout.write
    for k in xrange(w_h.shape[0]):
        if h[k] < 1e-2:
            write(".")
        elif h[k] < 0.5:
            write("-")
        elif h[k] < 1-(1e-2):
            write("+")
        else:
            write("*")
    print ""
    
