#!/bin/env python

"""
RSM training on documents.
$Id: rsm.py,v 1.8 2013/06/30 02:11:26 daichi Exp $

"""
import re
import sys
import gzip
import getopt
import pickle
import fmatrix
import rsm_numpy
import numpy as np

# default parameters
hiddens = 50
epochs = 1
iter = 1
rate = 0.001
batch = 1
proto = 1	# binary pickled model

def usage():
    print 'rsm.py, modified python implementation of Replicated Softmax Model.'
    print '$Id: rsm.py,v 1.8 2013/06/30 02:11:26 daichi Exp $'
    print 'usage  : rsm.py [options] train model'
    print 'options: -H hiddens number of hidden variables (default = %d)' % hiddens
    print '         -N epochs  number of learning epochs (default = %d)' % epochs
    print '         -n iter    iterations of contrastive divergence (default = %d)' %iter
    print '         -b batch   number of batch size (default = %d)' % batch
    print '         -r rate    learning rate (default = %g)' % rate
    sys.exit (0)

def main():
    shortopts = "H:N:n:r:b:h"
    longopts = ['hiddens=','epochs=','iter=','rate=','batch=','help']
    global hiddens, epochs, iter, rate, batch, proto
    try:
        opts,args = getopt.getopt(sys.argv[1:], shortopts, longopts)
    except getopt.GetoptError, err:
        usage ()
        
    # parse arguments
    for o, a in opts:
        if o in ('-H', '--hiddens'):
            hiddens = int(a)
        elif o in ('-N', '--epochs'):
            epochs = int(a)
        elif o in ('-n', '--iter'):
            iter = int(a)
        elif o in ('-r', '--rate'):
            rate = float(a)
        elif o in ('-b', '--batch'):
            batch = int(a)
        elif o in ('-h', '--help'):
            usage ()
        else:
            assert False, "unknown option"

    if len(args) == 2:
        train = args[0]
        model = args[1]
    else:
        usage ()

    print 'loading data..',; sys.stdout.flush()
    if re.search('\.pkz$', train):
        file = gzip.open (train, 'r')
        X = pickle.load(file)
        file.close ()
    else:
        X = fmatrix.parse(train)
    print 'done.'

    print 'number of documents        = %d' % X.shape[0]
    print 'number of lexicon          = %d' % X.shape[1]
    print 'number of hidden variables = %d' % hiddens
    print 'number of learning epochs  = %d' % epochs
    print 'number of CD iterations    = %d' % iter
    print 'minibatch size             = %d' % batch
    print 'learning rate              = %g' % rate
        
    RSM = rsm_numpy.RSM()
    result = RSM.train(X, hiddens, epochs, iter, lr=rate, btsz=batch)

    with open(model, 'wb') as file:
        pickle.dump (result, file, proto)

if __name__ == '__main__':
    main ()

