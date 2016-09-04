"""Do some things that are a bit like variable binding"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

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
flags.DEFINE_integer('dynamic_iterations', 0, 'whether to unroll the whole thing')
flags.DEFINE_float('decay', 1.0, 'how much to decay the learning rate')
flags.DEFINE_integer('decay_steps', 5000, 'how often to decay the learning rate')

flags.DEFINE_string('results_dir', None, 'Where to put the resuts')

#problem stuff
flags.DEFINE_string('task', 'recall', 'what to do')
flags.DEFINE_integer('num_items', 1, 'how many things to remember')
flags.DEFINE_integer('dimensionality', 8, 'size of patterns')
flags.DEFINE_integer('offset', 0, '1 or 0, whether to try remember the'
                     'current pattern or the following one')

FLAGS = flags.FLAGS


def get_cell():
    """Gets a cell"""
    if FLAGS.cell == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(
            FLAGS.width, state_is_tuple=True)
    if FLAGS.cell == 'vanilla':
        return tf.nn.rnn_cell.BasicRNNCell(FLAGS.width)
    if FLAGS.cell == 'cp-gate':
        return mrnn.CPGateCell(FLAGS.width, FLAGS.rank, candidate_nonlin=tf.nn.relu)
    if FLAGS.cell == 'cp-gate-combined':
        return mrnn.CPGateCell(FLAGS.width, FLAGS.rank, separate_pad=False)
    if FLAGS.cell == 'gru':
        return tf.nn.rnn_cell.GRUCell(FLAGS.width)
    else:
        raise ValueError('I do not know this: {}'.format(FLAGS.cell))


def image_summarise(data, tag):
    """Makes image summaries of a seq_len list of batch_size x features
    tensors"""
    image_data = tf.pack(data)
    image_data = tf.transpose(image_data, [1, 2, 0])
    image_data = tf.expand_dims(image_data, -1)
    tf.image_summary(tag, image_data)


def main(_):
    """do stuff"""
    os.makedirs(FLAGS.results_dir, exist_ok=True)

    with tf.variable_scope('inputs'):
        inputs, targets = data.get_recognition_tensors(
            FLAGS.batch_size, FLAGS.sequence_length, FLAGS.num_items,
            FLAGS.dimensionality, FLAGS.task, FLAGS.offset,
            inbetween_noise=False)
        inputs = tf.unpack(inputs)

    print('{:-^60}'.format('getting model'), end='', flush=True)
    with tf.variable_scope('model', initializer=mrnn.init.orthonormal_init()):
        cell = get_cell()
        if FLAGS.task == 'recall':
            num_outputs = FLAGS.dimensionality
        elif FLAGS.task == 'order':
            num_outputs = FLAGS.num_items
        _, _, logits, outputs = sm.inference(
            inputs, 1, cell, num_outputs, do_projection=False,
            full_logits=True, dynamic_iterations=FLAGS.dynamic_iterations)
        image_summarise([tf.nn.softmax(logit) for logit in logits], 
                        'output')
        image_summarise(outputs, 'states')
        image_summarise(inputs, 'inputs')
    print('\r{:~^60}'.format('got model'))

    print('{:-^60}'.format('getting training ops'), end='', flush=True)
    with tf.variable_scope('training'):
        if FLAGS.task == 'recall':
            loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits[-1], targets))
        elif FLAGS.task == 'order':
            loss_op = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits[-1], targets))
        else:
            raise ValueError('unknown task {}'.format(FLAGS.task))
        
        tf.scalar_summary('loss', loss_op)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if FLAGS.decay != 1.0:
            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate, global_step, FLAGS.decay_steps,
                FLAGS.decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate

        opt = tf.train.AdamOptimizer(learning_rate)
        train_op = opt.minimize(loss_op, global_step=global_step)
    print('\r{:~^60}'.format('got train ops'))
    
    all_summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.results_dir)
    sess = tf.Session()

    print('{:-^60}'.format('initialising'), end='', flush=True)
    sess.run(tf.initialize_all_variables())
    print('\r{:~^60}'.format('initialised'))

    for step in range(FLAGS.num_steps):

        loss, _ = sess.run([loss_op, train_op])

        if step % 10 == 0:
            print('\r({}) loss: {}'.format(step, loss), end='')
            summs = sess.run(all_summaries)
            writer.add_summary(summs, global_step=step)
        if step % FLAGS.decay_steps == 0:
            if type(learning_rate) is not float:
                print('\nlr: {}'.format(sess.run(learning_rate)))


if __name__ == '__main__':
    tf.app.run()