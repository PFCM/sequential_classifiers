"""Try some (potentially quite small) models on addition task.

Regression is just classification with lots of classes right?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import progressbar

import mrnn
import rnndatasets.addition as data

import sequential_model as sm

flags = tf.app.flags

flags.DEFINE_integer('width', 100, 'how wide should the recurrent layers be')
flags.DEFINE_integer('layers', 1, 'how many recurrent layers should there be')
flags.DEFINE_integer('num_epochs', 100, 'how long to train for')
flags.DEFINE_string('results_dir', 'time',
                    'where to store the results. If `time` a directory is '
                    'chosen based on the current time')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate for SGD')
flags.DEFINE_integer('batch_size', 100, 'how many examples to use for SGD')
flags.DEFINE_integer('rank', 10, 'the rank of the tensor decompositions')
flags.DEFINE_integer('sequence_length', 150, 'length of sequences')

flags.DEFINE_string(
    'cell',
    'lstm',
    'which cell to use. One of: `vanilla`, `cp-relu`, `cp-tanh`, `tt-relu`, '
    '`tt-tanh`, `irnn` or `lstm`.')
flags.DEFINE_float('max_grad_norm', 10.0,
                   'where to clip the global norm of the gradients during '
                   'backprop')

FLAGS = flags.FLAGS


def get_cell(size):
    """Gets an appropriate cell according to the flags.
    At the moment assumes you want input size = size.
    """
    if FLAGS.cell == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(size)  # default forget biases
    if FLAGS.cell == 'vanilla':
        return mrnn.VRNNCell(
            size, hh_init=mrnn.init.spectral_normalised_init(0.999))
    if FLAGS.cell == 'irnn':
        return mrnn.IRNNCell(size)
    if FLAGS.cell == 'cp-relu':
        return mrnn.SimpleCPCell(size, size, FLAGS.rank,
                                 nonlinearity=tf.nn.relu,
                                 weightnorm=False,
                                 separate_pad=True)
    if FLAGS.cell == 'cp-tanh':
        return mrnn.SimpleCPCell(size, size, FLAGS.rank,
                                 nonlinearity=tf.nn.tanh,
                                 weightnorm=False,
                                 separate_pad=True)
    if FLAGS.cell == 'tt-relu':
        return mrnn.SimpleTTCell(size, size, [FLAGS.rank]*2,
                                 nonlinearity=tf.nn.relu)
    if FLAGS.cell == 'tt-tanh':
        return mrnn.SimpleTTCell(size, size, [FLAGS.rank]*2,
                                 nonlinearity=tf.nn.tanh)
    if FLAGS.cell == 'cp+':
        return mrnn.AdditiveCPCell(
            size, size, FLAGS.rank, nonlinearity=tf.nn.tanh)
    if FLAGS.cell == 'cp+-':
        return mrnn.AddSubCPCell(
            size, size, FLAGS.rank, nonlinearity=tf.nn.relu)
    if FLAGS.cell == 'cp-del':
        return mrnn.CPDeltaCell(size, size, FLAGS.rank)
    if FLAGS.cell == 'vanilla-layernorm':
        return mrnn.VRNNCell(size, size, hh_init=mrnn.init.orthonormal_init(),
                             nonlinearity=tf.nn.tanh, weightnorm='layer')
    raise ValueError('Unknown cell: {}'.format(FLAGS.cell))


def mse(a, b):
    """returns mse between the two inputs"""
    return tf.reduce_mean(tf.squared_difference(a, b))


def main(_):
    """train a model"""
    results_file = os.path.join(FLAGS.results_dir, 'results.txt')
    os.makedirs(FLAGS.results_dir, exist_ok=True)

    train_data, test_data = data.get_data_batches(
        FLAGS.sequence_length, FLAGS.batch_size, 100000, 10000,
        num_epochs=FLAGS.num_epochs)

    train_inputs = tf.unpack(train_data[0])
    train_targets = train_data[1]
    test_inputs = tf.unpack(test_data[0])
    test_targets = test_data[1]
    with tf.variable_scope('model') as scope:
        cell = get_cell(FLAGS.width)
        # get a model with one output which we will leave linear
        _, _, logits, _ = sm.inference(
            train_inputs, FLAGS.layers, cell, 1)
        train_loss = mse(logits, train_targets)

        scope.reuse_variables()
        _, _, test_logits, _ = sm.inference(
            test_inputs, FLAGS.layers, cell, 1)
        test_loss = mse(test_logits, train_targets)

    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('train'):
        train_op, gnorm = sm.train(train_loss, FLAGS.learning_rate,
                                   global_step)

    # should be ready to go
    sess = tf.Session()
    print('initialising', end='', flush=True)
    sess.run(tf.initialize_all_variables())
    print('\rinitialised    ')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():

            train_batch_loss, grad_norm, _ = sess.run(
                [train_loss, gnorm, train_op])
            step = global_step.eval(session=sess)
            print('\r{} train loss (one batch): {:.4f} (grad norm: {:.4f})'.format(
                step, train_batch_loss, grad_norm), end='', flush=True)
            if step % 1000 == 0:  # arbitrary
                mse_sum, valid_steps = 0, 0
                for _ in range(10000 // FLAGS.batch_size):
                    valid_batch_loss = sess.run(test_loss)
                    valid_steps += 1
                    mse_sum += valid_batch_loss
                print('\n{} valid loss: {}'.format(step, mse_sum / valid_steps))
                with open(results_file, 'a') as fp:
                    fp.write('{}\n'.format(mse_sum / valid_steps))
    except tf.errors.OutOfRangeError:
        print('Done')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
