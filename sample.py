#!/usr/bin/env python
'''Extract a random sample from an HDF5 file.'''

import sys
import h5py
import numpy as np
import random

def extract_sample(dataset, count, outset):
    sets = dataset.keys()
    num = dataset[sets[0]].shape[-1]

    for name in sets:
        assert dataset[name].shape[-1] == num

    indices = sorted(random.sample(xrange(num), count))

    for name, dataset in dataset.iteritems():
        if len(dataset.shape) == 2:
            d = dataset[:,indices]
        else:
            d = dataset[:,:,indices]
        outset.create_dataset(name, data=d)


if __name__ == '__main__':
    infile, count, outfile = sys.argv[1:]
    count = int(count)
    f = h5py.File(infile, 'r')
    o = h5py.File(outfile, 'w')
    extract_sample(f, count, o)

