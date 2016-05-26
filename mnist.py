"""Do sequential mnist"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mrnn
import rnndatasets.sequentialmnist as data

import sequential_model as sm

flags = tf.app.flags

flags.DEFINE_integer('width', 100, 'how wide should the recurrent layers be')
flags.DEFINE_integer('layers', 1, 'how many recurrent layers should there be')
flags.DEFINE_bool('project', False, 'If true, adds a projection layer.')
flags.DEFINE_integer('num_epochs', 100, 'how long to train for')

FLAGS = flags.FLAGS


def run_epoch(sess, batch_iter, inputs, targets, train_op, loss):
    """Runs an epoch of training (or not training if train_op is tf.no_op()).
    Returns average loss for the epoch"""
    total_loss = 0
    num_steps = 0

    for data, labels in batch_iter:
        feed = {inputs[i]: data[i, ...] for i in range(len(inputs))}
        feed[targets] = labels

        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict=feed)
        total_loss += batch_loss
        num_steps += 1
        print('\r batch loss {:.5f}'.format(batch_loss),
              end='')
    return total_loss / num_steps


def count_params():
    """does a simple count of how many things are in tf.trainable_variables"""
    total = 0
    for var in tf.trainable_variables():
        prod = 1
        for dim in var.get_shape():
            prod *= dim.value
        total += prod
    return total


def main(_):
    # now we get the stuff
    # unfortunately we are going to have to do some serious
    # unrolling of the network.
    seq_length = 28*28  # how many mnist pixels
    batch_size = 100
    inputs = [tf.placeholder(tf.float32, name='input_{}'.format(i),
                             shape=[batch_size, 1])
              for i in range(seq_length)]
    targets = tf.placeholder(tf.int64, name='targets',
                             shape=[batch_size])

    learning_rate = tf.Variable(0.01)

    print('{:.^30}'.format('getting model'), end='', flush=True)
    with tf.variable_scope('model'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.width)
        init_state, final_state, logits = sm.inference(
            inputs, FLAGS.layers, cell, 10)
        loss = sm.loss(logits, targets)
        train_op = sm.train(loss, learning_rate)
        accuracy = sm.accuracy(logits, targets)
    print('\r{:\\^30}'.format('got model with {} params'.format(count_params())))

    sess = tf.Session()
    print('{:.^30}'.format('initialising'), end='', flush=True)
    sess.run(tf.initialize_all_variables())
    print('\r{:/^30}'.format('initialised'))

    print('{:.^30}'.format('getting data'), end='', flush=True)
    _, _, test = data.get_iters(batch_size)
    print('\r{:\\^30}'.format('got data'))

    for epoch in range(FLAGS.num_epochs):
        train, valid, _ = data.get_iters(batch_size)
        # do a training run
        print('{:/<20}'.format('Epoch {}'.format(epoch+1)))
        train_loss = run_epoch(sess, train, inputs, targets,
                               train_op, loss)
        print()
        valid_accuracy = run_epoch(sess, valid, inputs, targets,
                                   tf.no_op(), accuracy)
        print()
        print('---Train loss:     {}'.format(train_loss))
        print('---Valid accuracy: {}'.format(valid_accuracy))


    print('{:\\^30}'.format('testing'))
    test_accuracy = run_epoch(sess, test, inputs, targets,
                              tf.no_op(), accuracy)
    print('{:^30}'.format(test_accuracy))


if __name__ == '__main__':
    tf.app.run()
