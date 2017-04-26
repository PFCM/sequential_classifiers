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
import rnndatasets.synthetic.addition as data

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
flags.DEFINE_bool('online', False, 'Whether to generate training/test data'
                  ' or just generate batches of examples one at a time as'
                  ' required')
flags.DEFINE_string('optimiser', 'adam', 'adam, rmsprop or momentum')
flags.DEFINE_bool('save', True, 'Whether or not to save actual model files')

FLAGS = flags.FLAGS


def get_cell(size):
    """Gets an appropriate cell according to the flags.
    At the moment assumes you want input size = size.
    """
    if FLAGS.cell == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)  # default forget biases
    if FLAGS.cell == 'vanilla':
        return mrnn.VRNNCell(
            size, hh_init=mrnn.init.orthonormal_init())
    if FLAGS.cell == 'irnn':
        return mrnn.IRNNCell(size, 2)
    if FLAGS.cell == 'cp-relu':
        return mrnn.SimpleCPCell(size, size, FLAGS.rank,
                                 nonlinearity=tf.nn.relu,
                                 weightnorm=False,
                                 separate_pad=True)
    if FLAGS.cell == 'cp-tanh':
        return mrnn.SimpleCPCell(size, 2, FLAGS.rank,
                                 nonlinearity=tf.nn.tanh,
                                 weightnorm=False,
                                 separate_pad=True)
    if FLAGS.cell == 'tt-relu':
        return mrnn.SimpleTTCell(size, 2, [FLAGS.rank]*2,
                                 nonlinearity=tf.nn.relu)
    if FLAGS.cell == 'tt-tanh':
        return mrnn.SimpleTTCell(size, 2, [FLAGS.rank]*2,
                                 nonlinearity=tf.nn.tanh)
    if FLAGS.cell == 'cp+':
        return mrnn.AdditiveCPCell(
            size, size, FLAGS.rank, nonlinearity=tf.nn.tanh)
    if FLAGS.cell == 'cp+-':
        return mrnn.AddSubCPCell(
            size, size, FLAGS.rank, nonlinearity=tf.nn.relu)
    if FLAGS.cell == 'cp-del':
        return mrnn.CPDeltaCell(size, 2, FLAGS.rank)
    if FLAGS.cell == 'cp-gate':
        return mrnn.CPGateCell(size, FLAGS.rank)
    if FLAGS.cell == 'cp-gate-combined':
        return mrnn.CPGateCell(size, FLAGS.rank, separate_pad=False, candidate_nonlin=tf.identity)
    if FLAGS.cell == 'vanilla-layernorm':
        return mrnn.VRNNCell(size, 2, hh_init=mrnn.init.orthonormal_init(),
                             nonlinearity=tf.nn.tanh, weightnorm='layer')
    if FLAGS.cell == 'gru':
        return tf.nn.rnn_cell.GRUCell(size)
    raise ValueError('Unknown cell: {}'.format(FLAGS.cell))


def mse(a, b, summary_name=None):
    """returns mse between the two inputs"""
    loss = tf.reduce_mean(tf.squared_difference(a, b))
    if summary_name:
        tf.scalar_summary(summary_name, loss)
    return loss


def main(_):
    """train a model"""
    results_file = os.path.join(FLAGS.results_dir, 'results.txt')
    os.makedirs(FLAGS.results_dir, exist_ok=True)

    if FLAGS.online:
        train_inputs, train_targets = data.get_online_sequences(
            FLAGS.sequence_length, FLAGS.batch_size)
        train_inputs = tf.train.limit_epochs(train_inputs,
                                             num_epochs=FLAGS.num_epochs)
        train_inputs = tf.unpack(train_inputs)
    else:
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
            train_inputs, FLAGS.layers, cell, 1, do_projection=False,
            dynamic_iterations=256)
        logits = tf.squeeze(logits)
        train_loss = mse(logits, train_targets, 'train mse')
        if not FLAGS.online:
            scope.reuse_variables()
            _, _, test_logits, _ = sm.inference(
                test_inputs, FLAGS.layers, cell, 1, do_projection=False,
                dynamic_iterations=128)
            test_logits = tf.squeeze(test_logits)
            test_loss = mse(test_logits, test_targets, 'test mse')

    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.variable_scope('train'):
        train_op, gnorm = sm.train(train_loss, FLAGS.learning_rate,
                                   global_step, optimiser=FLAGS.optimiser)
        tf.scalar_summary('gradient norm', gnorm)

    # should be ready to go
    sess = tf.Session()
    print('initialising', end='', flush=True)
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    print('\rinitialised    ')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summ_writer = tf.train.SummaryWriter(FLAGS.results_dir, graph=sess.graph)
    all_summs = tf.merge_all_summaries()

    # how many times are we going to go?
    if FLAGS.online:
        num_steps = FLAGS.num_epochs
    else:
        num_steps = (100000 // FLAGS.batch_size) * FLAGS.num_epochs
    bar = progressbar.ProgressBar(
        widgets=['[', progressbar.Counter(), '] ',
                 '(｢๑•₃•)｢ ',
                 progressbar.Bar(marker='=', left='', right='☰'),
                 ' (', progressbar.DynamicMessage('loss'), ')',
                 '{', progressbar.ETA(), '}'],
        redirect_stdout=True)

    try:
        bar.start(num_steps)
        gnorm_sum = 0
        tloss_sum = 0
        tloss = 0
        while not coord.should_stop() or FLAGS.online:

            train_batch_loss, grad_norm, _ = sess.run(
                [train_loss, gnorm, train_op])
            step = global_step.eval(session=sess)
            gnorm_sum += grad_norm
            tloss_sum += train_batch_loss
            tloss += train_batch_loss
            if step % 10 == 0:  # don't waste too much time looking pretty
                bar.update(step, loss=tloss_sum/10)
                tloss_sum = 0
                summaries = sess.run(all_summs)
                summ_writer.add_summary(summaries, global_step=step)
                # inputs, targets, outputs, tloss = sess.run([train_data[0], train_targets, logits, train_loss])
                # print('in: {}'.format(inputs))
                # print('t: {}'.format(targets))
                # print('o: {}'.format(outputs))
                # print('mse: {}'.format(tloss))
                # return
            if step % 10 == 0:  # arbitrary
                if FLAGS.online:
                    mse_sum = tloss
                    tloss = 0
                    valid_steps = 1000
                else:
                    mse_sum, valid_steps = 0, 0
                    for _ in range(10000 // FLAGS.batch_size):
                        valid_batch_loss = sess.run(test_loss)
                        valid_steps += 1
                        mse_sum += valid_batch_loss
                    print('\n{} valid loss: {}, avge grad norm: {}'.format(
                        step, mse_sum / valid_steps, gnorm_sum / 1000))

                with open(results_file, 'a') as fp:
                    fp.write('{}, {}\n'.format
                             (mse_sum / valid_steps, gnorm_sum/1000))
                gnorm_sum = 0
    except tf.errors.OutOfRangeError:
        print('Done')
        if FLAGS.save:
            print('..saving..', end='')
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(sess, os.path.join(FLAGS.results_dir, 'model'),
                       write_meta_graph=False)
            print('\r  saved  ')

    finally:
        coord.request_stop()
        bar.finish()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
