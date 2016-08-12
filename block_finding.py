"""This task is to classify sequences as either having or not having a large block
of ones in a row."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import progressbar

import mrnn
from rnndatasets.template import online_block_tensors

import sequential_model as sm

flags = tf.app.flags

flags.DEFINE_integer('width', 100, 'number of hidden units')
flags.DEFINE_integer('num_steps', 100000, 'number of parameter updates')
flags.DEFINE_string('results_dir', None, 'where to put results')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate for SGD')
flags.DEFINE_integer('batch_size', 32, 'how many at a time')
flags.DEFINE_integer('sequence_length', 100, 'how hard')
flags.DEFINE_integer('block_size', 4, 'size of chunks to find')
flags.DEFINE_float('one_prob', 0.3, 'quantity of noise')
flags.DEFINE_string('cell', 'vanilla', 'type of rnn')
flags.DEFINE_float('max_grad_norm', 100000.0, 'where to clip grads')
flags.DEFINE_bool('write_graph', True, 'whether to write a graph summary')
flags.DEFINE_integer('summarise_every', 10, 'how often to write summaries, print')
flags.DEFINE_integer('rank', 50, 'rank of tensor decompositions')
flags.DEFINE_bool('fixed_length', False, 'whether sequences have random lengths')

FLAGS = flags.FLAGS


def get_cell():
    """gets cell according to flags"""
    if FLAGS.cell == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(FLAGS.width, state_is_tuple=True)
    if FLAGS.cell == 'vanilla':
        return mrnn.VRNNCell(FLAGS.width, input_size=1,
                             hh_init=mrnn.init.orthonormal_init())
    if FLAGS.cell == 'gru':
        return tf.nn.rnn_cell.GRUCell(FLAGS.width)
    if FLAGS.cell == 'cp-tanh':
        return mrnn.SimpleCPCell(FLAGS.width, 1, FLAGS.rank,
                                 nonlinearity=tf.nn.tanh,
                                 weightnorm=False,
                                 separate_pad=True)
    if FLAGS.cell == 'cp+':
        return mrnn.AdditiveCPCell(
            FLAGS.width, 1, FLAGS.rank, nonlinearity=tf.nn.relu)
    if FLAGS.cell == 'cp-gate':
        return mrnn.CPGateCell(FLAGS.width, FLAGS.rank)
    raise ValueError('Unknown cell: {}'.format(FLAGS.cell))


def run_training(train_op, loss_op, gnorm, global_step, logits, labels, inputs):
    """actuall does the stuff"""

    sess = tf.Session()
    print('{:_^40}'.format('initialising'), end='')
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    print('\r{:~^40}'.format('initialised'))
    
    writer = tf.train.SummaryWriter(FLAGS.results_dir)
    if FLAGS.write_graph:
        writer.add_graph(sess.graph)

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        all_summaries = tf.merge_all_summaries()

        bar = progressbar.ProgressBar(
            widgets=['[', progressbar.Counter(), '] ',
                     '\(x₃x)/',
                     progressbar.Bar(marker='/'),
                     ' (', progressbar.DynamicMessage('loss'), ') ',
                     '[', progressbar.ETA(), ']'],
            redirect_stdout=True)

        try:
            bar.start(FLAGS.num_steps)
            while True:
                batch_loss, _ = sess.run([loss_op, train_op])

                #debug_stuff = sess.run(inputs + [tf.nn.sigmoid(logits), labels, loss_op])
                #print(debug_stuff)
                #break
                
                step = global_step.eval()

                if step % FLAGS.summarise_every == 0:
                    bar.update(step, loss=batch_loss)
                    summaries = sess.run(all_summaries)
                    writer.add_summary(summaries, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done')
        finally:
            coord.request_stop()
            bar.finish()
        coord.join(threads)
        sess.close()


def main(_):
    os.makedirs(FLAGS.results_dir, exist_ok=True)

    global_step = tf.Variable(0, trainable=False)
    
    with tf.variable_scope('inputs'):
        inputs, lengths, labels = online_block_tensors(FLAGS.batch_size,
                                                       FLAGS.sequence_length,
                                                       FLAGS.block_size,
                                                       FLAGS.one_prob,
                                                       FLAGS.fixed_length)
        if FLAGS.fixed_length:
            lengths = None
        inputs = inputs * 2.0 - 1.0
        inputs = tf.train.limit_epochs(inputs, FLAGS.num_steps)
        inputs = tf.unpack(inputs)
        labels = tf.expand_dims(labels, 1)
    print('{:~^40}'.format('have data tensors'))

    with tf.variable_scope('rnn'):
        cell = get_cell()
        _, _, logits, _ = sm.inference(
            inputs, 1, cell, 1, do_projection=False,
            dynamic_iterations=0, lengths=lengths)
    print('{:~^40}'.format('have rnn'))
        
    with tf.variable_scope('train'):
        loss_op = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
        loss_op = tf.reduce_mean(loss_op)
        train_op, gnorm = sm.train(loss_op, FLAGS.learning_rate, global_step,
                                   FLAGS.max_grad_norm, optimiser='adam')
        tf.scalar_summary('xent', loss_op)
        tf.scalar_summary('gnorm', gnorm)
    print('{:~^40}'.format('have training ops'))

    with tf.variable_scope('eval'):
        # accuracy = tf.reduce_mean(
        #     tf.cast(tf.round(tf.nn.sigmoid(logits)) == labels, tf.float32))
        accuracy = tf.contrib.metrics.accuracy(
            tf.nn.sigmoid(logits) > 0.5, tf.cast(labels, tf.bool))
        tf.scalar_summary('accuracy', accuracy)

    # and now train
    run_training(train_op, loss_op, gnorm, global_step, logits, labels, inputs)
    

if __name__ == '__main__':
    tf.app.run()
