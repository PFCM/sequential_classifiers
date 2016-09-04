"""Do some things that are a bit like variable binding"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import progressbar

import mrnn
import rnndatasets.synthetic.binding as data

import sequential_model as sm

flags = tf.app.flags

flags.DEFINE_string('cell', 'gru', 'what cell to use')
flags.DEFINE_integer('width', 80, 'how many')
flags.DEFINE_integer('rank', 40, 'rank of tensor decomp')

flags.DEFINE_integer('num_steps', 10000, 'how long should we go?')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate for SGD')
flags.DEFINE_integer('batch_size', 50, 'how many to train on at once')
flags.DEFINE_integer('sequence_length', 200, 'how long to remember')
flags.DEFINE_float('max_grad_norm', 10000.0, 'how much clippery')

flags.DEFINE_string('results_dir', None, 'Where to put the resuts')

#problem stuff
flags.DEFINE_string('task', 'recall', 'what to do')
flags.DEFINE_integer('num_items', 1, 'how many things to remember')
flags.DEFINE_integer('dimensionality', 8, 'size of patterns')

FLAGS = flags.FLAGS


def get_cell():
    """Gets a cell"""
    if FLAGS.cell == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(
            FLAGS.width, state_is_tuple=True)
    if FLAGS.cell == 'vanilla':
        return mrnn.VRNNCell(FLAGS.width, hh_init=mrnn.init.orthonormal_init())
    if FLAGS.cell == 'cp-gate':
        return mrnn.CPGateCell(FLAGS.width, FLAGS.rank)
    if FLAGS.cell == 'gru':
        return tf.nn.rnn_cell.GRUCell(FLAGS.width)
    else:
        raise ValueError('I do not know this: {}'.format(FLAGS.cell))


def main(_):
    """do stuff"""

    with tf.variable_scope('inputs'):
        inputs, targets = data.get_recognition_tensors(
            FLAGS.batch_size, FLAGS.sequence_length, FLAGS.num_items,
            FLAGS.dimensionality, FLAGS.task)
        inputs = tf.unpack(inputs)

    print('{:-^60}'.format('getting model'), end='', flush=True)
    with tf.variable_scope('model'):
        cell = get_cell()
        if FLAGS.task == 'recall':
            num_outputs = FLAGS.dimensionality
        elif FLAGS.task == 'order':
            num_outputs = FLAGS.num_items + 1
        _, _, logits, _ = sm.inference(
            inputs, 1, cell, num_outputs, do_projection=False,
            full_logits=False, dynamic_iterations=0)
    print('\r{:~^60}'.format('got model'))

    print('{:-^60}'.format('getting training ops'), end='', flush=True)
    with tf.variable_scope('training'):
        if FLAGS.task == 'recall':
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits, targets))


        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = opt.minimize(loss_op)
    print('\r{:~^60}'.format('got train ops'))

    sess = tf.Session()

    print('{:-^60}'.format('initialising'), end='', flush=True)
    sess.run(tf.initialize_all_variables())
    print('\r{:~^60}'.format('initialised'))

    for step in range(FLAGS.num_steps):

        loss, _ = sess.run([loss_op, train_op])

        if step % 10 == 0:
            print('\r({}) loss: {}'.format(step, loss), end='')


if __name__ == '__main__':
    tf.app.run()
