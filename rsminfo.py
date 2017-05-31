#!/usr/local/bin/python

import sys
import gzip
import pickle

def rsminfo (file):
    with open (file, 'r') as fh:
        model = pickle.load (fh)
    print '[FILE %s]' % file
    print 'PPL     = %g' % model['ppl']
    print 'hidden  = %d' % model['w_h'].shape[0]
    print 'lexicon = %d' % model['w_v'].shape[0]
    print 'rate    = %g' % model['rate']
    print 'iter    = %d' % model['iter']
    print 'batch   = %d' % model['batch']
    print 'epochs  = %d' % model['epoch']
    print 'init    = %g' % model['init']
    print 'moment  = %g' % model['moment']

for file in sys.argv[1:]:
    rsminfo (file)
    
