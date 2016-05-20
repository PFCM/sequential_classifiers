"""Some bits and pieces for putting together a model for sequential mnist.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mrnn


def _recurrent_inference(inputs):
    """gets the forward pass of the recurrent part of the model.

    Args:
        inputs: placeholder for the input images. Expected to be
            of shape `[time_steps, batch_size, num_features]`.
    """
    pass


def _feedforward_inference(inputs):
    """Gets the forward part of the classification part of the model.

    Args:
        inputs: the input. Might be the final hidden state or the final
            output etc.
    """
    pass


def inference(inputs):
    """Gets the forward step of the model.

    Args:
        inputs: the input placeholder.

    """
    pass


def loss(logits):
    """Gets some kind of loss from the outputs of the classifier"""
    pass


def train(loss):
    """Gets an op to minimise the given loss"""
    pass
