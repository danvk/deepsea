#!/usr/bin/env python
# See http://stackoverflow.com/a/34952915/388951
import tensorflow as tf

def renorm(x, axis, max_norm):
    '''Renormalizes the sub-tensors along axis such that they do not exceed norm max_norm.'''
    # This elaborate dance avoids empty slices, which TF dislikes.
    rank = tf.rank(x)
    bigrange = tf.range(-1, rank + 1)
    dims = tf.slice(
                tf.concat(0, [tf.slice(bigrange, [0], [1 + axis]),
                              tf.slice(bigrange, [axis + 2], [-1])]),
                [1], rank - [1])

    # Determine which columns need to be renormalized.
    l2norm_inv = tf.rsqrt(tf.reduce_sum(x * x, dims, keep_dims=True))
    scale = max_norm * tf.minimum(l2norm_inv, tf.constant(1.0 / max_norm))

    # Broadcast the scalings
    return tf.mul(scale, x)


if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.constant([0., 0., 3., 4., 30., 40., 300., 400.], shape=(4, 2))

        print x.eval()
        print renorm(x, 0, 10).eval()
        print renorm(x, 1, 350).eval()
