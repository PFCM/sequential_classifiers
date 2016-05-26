"""Do sequential mnist"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mrnn
import rnndatasets.sequentialmnist as sm

flags = tf.app.flags

flags.DEFINE_integer('width', 256, 'how wide should the recurrent layers be')
flags.DEFINE_integer('layer', 1, 'how many recurrent layers should there be')
flags.DEFINE_bool('project', False, 'If true, adds a projection layer.')

FLAGS = flags.FLAGS


def chop_sequences(data, seq_length):
    """Chops a batch of long sequences into smaller sentences.

    Args:
        data: a batch of sequences (numpy array). We assume it is of shape
        seq_length: the maximum length of the sequences yielded.
            This should divide the length of the actual sequences
            or there will be trouble. Well, missing values.

    Yields:
        batches: sequential bits of sequences (what does that mean).
    """
    long_seq = data.shape[1]
    num_chunks = long_seq // seq_length
    for i in range(num_chunks):
        yield data[i*seq_length:(i+1)*seq_length, ...]


def main(_):
    # now we get the stuff
    print('hi')


if __name__ == '__main__':
    tf.app.run()
