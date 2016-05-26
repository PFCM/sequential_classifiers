"""Some bits and pieces for putting together a model for sequential mnist.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mrnn


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


def inference(inputs, cell='lstm'):
    """Gets the forward step of the model.

    Args:
        inputs: a list of input placeholders, each of shape
            `[batch_size, num_inputs]`
        cell: what cell to use
    """
    pass


def loss(logits, targets):
    """Gets some kind of loss from the outputs of the classifier"""
    pass


def train(loss):
    """Gets an op to minimise the given loss"""
    pass
