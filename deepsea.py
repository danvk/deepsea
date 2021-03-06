#!/usr/bin/env python

'''Train a TensorFlow model on the DeepSEA data set.

See
http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html
https://docs.google.com/document/d/1-PrZB7IcNiADFU84a0H4AcUxpcPCq3PXev1SQPsMwXE/edit
https://docs.google.com/document/d/1dp2Kn7258Tttd_-FEnM05HOUJzevq9B9GJn2lngcrfo/edit#
'''

import h5py
import tensorflow as tf
import numpy as np
import random

FLAGS = tf.app.flags.FLAGS

def seq_to_str(seq):
    assert seq.shape[1] == 4
    def eq(a, b):
        return (a == b).all()
    def unhot(locus):
        if eq(locus, [1, 0, 0, 0]):
            return 'A'
        elif eq(locus, [0, 1, 0, 0]):
            return 'G'
        elif eq(locus, [0, 0, 1, 0]):
            return 'C'
        elif eq(locus, [0, 0, 0, 1]):
            return 'T'
        elif eq(locus, [0, 0, 0, 0]):
            return 'N'
        raise ValueError('Invalid 1-hot encoding: %s' % locus)
    seq_str = ''.join(unhot(locus) for locus in seq)
    assert len(seq_str) == seq.shape[0]  # ensures that this is really a 1-hot encoding
    return seq_str


def load_data(path):
    '''Loads DeepSEA HDF5 data and reorders the axes.

    Returns (sequences, labels), where the first axis is sample #.
    '''
    f = h5py.File(path, 'r')
    sequences = f['trainxdata']
    labels = f['traindata']
    return np.rollaxis(sequences[:,:,:], 2), np.rollaxis(labels[:,:], 1)


def seqs(xs, ys, batch_size, num_epochs=1):
    '''Returns an iterator of (seq, label) pairs.'''
    num_seqs = xs.shape[0]

    return ((xs[i:i+batch_size,:,:], ys[i:i+batch_size,:])
            for epoch in xrange(0, num_epochs)
            for i in xrange(0, num_seqs, batch_size))


def split_train_test(xs, ys, test_count=1000):
    '''Split inputs & outputs into training & test.

    Returns (train_xs, train_ys), (test_xs, test_ys).
    '''
    return ((xs[test_count:,:,:], ys[test_count:,:]),
            (xs[:test_count,:,:], ys[:test_count,:]))


def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)


def bias_variable(shape):
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.constant(0.0, shape=shape)
    #initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def conv_pool_1d(x, num_kernels, input_channels, window_size, name='conv'):
    W = weight_variable([window_size, 1, input_channels, num_kernels])
    b = bias_variable([num_kernels])
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    h_conv = tf.nn.relu(conv + b)

    # for max pooling, use window size=4x1, step size=4
    y = tf.nn.max_pool(h_conv, ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1], padding='VALID')

    tf.histogram_summary(name + '_W', W)
    tf.histogram_summary(name + '_b', b)

    return y


def l2norm(Ws):
    return sum(tf.reduce_sum(tf.pow(W, 2)) for W in Ws)


def l1norm(Ws):
    return sum(tf.reduce_sum(tf.abs(W)) for W in Ws)


if __name__ == '__main__':
    # Build the Graph
    x = tf.placeholder('float', shape=[None, 1000, 4])  # inputs: 1-hot sequences
    y_ = tf.placeholder('float', shape=[None, 919])  # labels

    # reshape the input as a 4-channel, 1000x1 image
    x_channels = tf.reshape(x, [-1, 1000, 1, 4])

    # Input: 1000x1x4
    # Output: (1000-8+1)/4=248x1x320
    with tf.name_scope('conv1') as scope:
        h_pool1 = conv_pool_1d(x_channels,
                               input_channels=4,  # A, C, G, T
                               num_kernels=320,
                               window_size=8,
                               name='conv1')

    # Input: 248x1x320
    # Output: 60x1x480
    with tf.name_scope('conv2') as scope:
        h_pool2 = conv_pool_1d(h_pool1,
                               input_channels=320,
                               num_kernels=480,
                               window_size=8,
                               name='conv2')
    # Input: 60x1x480
    # Output: 13x1x960
    with tf.name_scope('conv3') as scope:
        h_pool3 = conv_pool_1d(h_pool2,
                               input_channels=480,
                               num_kernels=960,
                               window_size=8,
                               name='conv3')

    h_pool3_flat = tf.reshape(h_pool3, [-1, 960 * 13])
    
    # Fully-connected layer
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([960 * 13, 925])
        b_fc1 = bias_variable([925])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        tf.histogram_summary('fc1_W', W_fc1)
        tf.histogram_summary('fc1_b', b_fc1)
        tf.histogram_summary('fc1_h', h_fc1)
        tf.scalar_summary('fc1_norm_h', l1norm([h_fc1]))

    # Sigmoid output layer
    with tf.name_scope('sigmoid') as scope:
        W_fc2 = weight_variable([925, 919])
        b_fc2 = bias_variable([919])
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
        y = tf.nn.sigmoid(h_fc2)
        tf.histogram_summary('fc2_W', W_fc2)
        tf.histogram_summary('fc2_b', b_fc2)
        tf.histogram_summary('fc2_h', h_fc2)
        tf.scalar_summary('fc2_norm_h', l1norm([h_fc2]))
        tf.histogram_summary('y', y)

    # Error term: log-likelihood
    # TODO: regularization
    error = -tf.reduce_sum(y_*tf.log(y) + (1-y_)*tf.log(1-y))
    regularization = 5e-7 * l2norm([W_fc1, W_fc2]) + 1e-8 * l1norm([h_fc1])

    loss = error + regularization

    # TODO: track weights for convolutional layers as well.
    tf.scalar_summary('error_summary', error)

    merged = tf.merge_all_summaries()

    # Train using gradient descent
    # train_step = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9).minimize(error)
    # XXX: The DeepSEA code uses a learning rate of 1.0, which immediately diverges.
    train_step = tf.train.RMSPropOptimizer(learning_rate=1e-3, momentum=0.9, decay=8e-7).minimize(loss)

    # Load the data and split it into train/test: 99% train, 1% test.
    sequences, labels = load_data('train100k.mat')
    print 'Input: %s --> %s' % (sequences.shape, labels.shape)
    (train_xs, train_ys), (test_xs, test_ys) = split_train_test(sequences, labels)

    print 'Train: %s --> %s' % (train_xs.shape, train_ys.shape)
    print 'Test:  %s --> %s' % (test_xs.shape, test_ys.shape)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("/tmp/deepsea_logs", sess.graph_def)
        sess.run(tf.initialize_all_variables())
        print 'Initialized!'

        for i, (xs, labels) in enumerate(
                seqs(train_xs, train_ys, batch_size=16, num_epochs=10)):
            if i % 10 == 0:
                print '%d...' % i
            if i % 100 == 0:
                summary_str, nll = sess.run([merged, error], feed_dict={x: test_xs, y_: test_ys})
                writer.add_summary(summary_str, i)
                print 'step %5d training error=%g' % (i, nll)

            train_step.run(feed_dict={x: xs, y_: labels})

