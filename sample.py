#!/usr/bin/env python
'''Extract a random sample from an HDF5 file.

Usage:
    ./sample.py in.mat 12345 out.mat
'''

import sys
import h5py
import numpy as np
import random


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def extract_sample(dataset, count, outset):
    sets = dataset.keys()
    num = dataset[sets[0]].shape[-1]

    for name in sets:
        assert dataset[name].shape[-1] == num

    indices = sorted(random.sample(xrange(num), count))

    for name, dataset in dataset.iteritems():
        shape = list(dataset.shape)
        shape[-1] = count
        print shape

        o = outset.create_dataset(name, shape, dtype=dataset.dtype)

        b = 100  # chunk size
        for n, idxs in enumerate(chunks(indices, b)):
            if len(dataset.shape) == 2:
                o[:,n*b:(n+1)*b] = dataset[:,idxs]
            else:
                o[:,:,n*b:(n+1)*b] = dataset[:,:,idxs]
            print '%s %d/%d...' % (name, n * b, count)


if __name__ == '__main__':
    infile, count, outfile = sys.argv[1:]
    count = int(count)
    f = h5py.File(infile, 'r')
    o = h5py.File(outfile, 'w')
    extract_sample(f, count, o)

