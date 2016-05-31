"""Some bits and pieces for putting together a model for sequential mnist.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _recurrent_inference(inputs, rnn_cell):
    """gets the forward pass of the recurrent part of the model.

    Args:
        inputs: placeholder for the input images. Expected to be
            a list of tensors of shape `[batch_size, rnn_cell.input_size]`
        rnn_cell: the tf.nn.rnn_cell.RNNCell to construct the network with.
    Returns:
        A triple (zero_state, outputs, final_state) where zero_state is the
            variable for the initial state of the network, outputs is a list
            of the network's outputs (one for each input) and final_state is
            the state of the network at the end of the batch.
    """
    batch_size = inputs[0].get_shape()[0].value  # get the batch size
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.nn.rnn(rnn_cell, inputs,
                                     initial_state=init_state)
    return init_state, outputs, final_state


def _feedforward_inference(inputs, num_classes, scope=None):
    """Gets the forward part of the classification part of the model.
    For now this is just a projection + bias (intended for use with softmax)

    Args:
        inputs: the input. Might be the final hidden state or the final
            output etc. Whatever it is we expect this to be of shape
            `[batch_size, features]`.
        num_classes: how many logits we will return.
        scope: a scope in which to make the variables.

    Returns:
        logits: un-softmaxed classifier outputs.
    """
    input_size = inputs.get_shape()[1].value
    with tf.variable_scope(scope or 'softmax_projection'):
        weights = tf.get_variable('W', [input_size, num_classes],
                                  dtype=tf.float32)
        bias = tf.get_variable('b', [num_classes], dtype=tf.float32)
        logits = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
        return logits


def inference(inputs, num_layers,
              cell, num_classes,
              do_projection=True,
              classify_state=False):
    """Gets the forward step of the model.

    Optionally adds a projection layer, which is just a matmul
    (linear layer with no bias).

    Args:
        inputs: a list of input placeholders, each of shape
            `[batch_size, num_inputs]`
        width: the width (number of hidden units) of the recurrent
            layers.
        num_layers: how many recurrent layers.
        cell: the raw cell for a single layer. If do_projection is
            false then it's up to you to make sure the sizes match up
            (ie. set num_layers to 1 and make the MultiRNNCell by hand).
        num_classes: how many classes we are going to try and classify it
            into.
        do_projection: whether or not to add a projection layer.
        classify_state: whether the input to the classification part
            is the final state of the model or the final output.

        Returns:
            triple of (initial_state, final_state, logits). The first and
                last states of the network are returned because these variables
                can be essential for training (especially on longer sequences
                where we have to significantly truncate the backprop through
                time). The last one is the raw classifier output, to actually
                predict probabilities, softmax this.

    """
    in_size = inputs[0].get_shape()[1].value
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    if do_projection:
        with tf.variable_scope('input_projection'):
            proj = tf.get_variable('projection', [in_size, cell.output_size])
            inputs = [tf.matmul(input_, proj) for input_ in inputs]

    initial_state, outputs, final_state = _recurrent_inference(inputs, cell)
    if classify_state:
        logits = _feedforward_inference(final_state, num_classes)
    else:
        logits = _feedforward_inference(outputs[-1], num_classes)
    return initial_state, final_state, logits


def loss(logits, targets):
    """Gets some kind of loss from the outputs of the classifier.
    Specifically, the softmax cross entropy. `targets` can be single ints
    (which should be a bit quicker) or actual distributions (in which case
    multiple labels are supported).

    Args:
        logits: the raw (unscaled) logits `[batch_size, num_classes]`
        targets: either `[batch_size, 1]` and dtype int64, otherwise
            `[batch_size, num_classes]` and some kind of float.

    Returns:
        a tensor with the cross entropy, averaged over the instances in the
           batch.
    """
    if len(targets.get_shape()) == 1:
        batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, targets)
    else:
        batch_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits, targets)
    return tf.reduce_mean(batch_losses)


def train(loss, learning_rate, global_step, grad_norm=10.0):
    """Gets an op to minimise the given loss"""
    # opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    tvars = tf.trainable_variables()
    grads = opt.compute_gradients(loss, tvars)
    grads, norm = tf.clip_by_global_norm([grad for grad, _ in grads], grad_norm)
    return opt.apply_gradients(zip(grads, tvars), global_step=global_step), norm


def accuracy(logits, targets):
    """Gets an op that tells you how accurate you are over a batch"""
    # we are just taking arg max so should be no need to softmax?
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, targets, 1), tf.float32))
