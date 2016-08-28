"""
Try and have a look at what the addition gents are actually up to.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import mrnn
import rnndatasets.addition as data

import sequential_model as sm
import addition


FLAGS = tf.app.flags.FLAGS


def get_png_encoders(images):
    """Gets ops to encode a tensor of shape `[time x batch x state_size]` into a batch of
    pictures"""
    images = tf.transpose(images, [1, 2, 0])  # shuffle them to [batch x state x time]
    images = tf.expand_dims(images, 3)  # add a channel dim of 1 for grayscale

    # normalise the images between 0-255
    # more or less with the same process as tf.image_summary

    def _negative_normalise():
        # rescale so that either the smallest value is -127 or the largest is 127
        smallest = tf.reduce_min(images)
        largest = tf.reduce_max(images)
        factor = tf.minimum(-127 / smallest, 127 / largest)
        # apply, and shift
        return factor * images + 127

    def _positive_normalise():
        # just rescale so the largest value is 255
        return images * (255 / tf.reduce_max(images))

    images = tf.cond(tf.reduce_any(images < 0),
                     _negative_normalise, _positive_normalise)
    images = tf.saturate_cast(images, tf.uint8)

    # unpack along the batch dimension, encode and repack
    return tf.pack(
        [tf.image.encode_png(im) for im in tf.unpack(images)])


def write_images(images, name):
    """Gets a numpy array of appropriate bytes and writes each to a file"""
    print(images.shape)
    for i, image in enumerate(images):
        img_name = os.path.join(name + '{}.png'.format(i))
        with open(img_name, 'wb') as fp:
            fp.write(bytes(image))


if __name__ == '__main__':
    # grab a batch of data, spit out some pics. easy.
    # (unless we decide it is necessary to directly inspect the gates,
    # which it probably is).

    inputs, targets = data.get_online_sequences(
        FLAGS.sequence_length,
        FLAGS.batch_size)
    inputs = tf.unpack(inputs)

    # do this the same as addition.py
    with tf.variable_scope('model') as scope:
        cell = addition.get_cell(FLAGS.width)
        init_state, final_state, logits, outputs = sm.inference(
            inputs, FLAGS.layers, cell, 1, do_projection=False, full_logits=True)
        print('got model')

    pic_dir = os.path.join(FLAGS.results_dir, 'pics')
    os.makedirs(pic_dir, exist_ok=True)

    output_image_bytes = get_png_encoders(tf.pack(outputs))
    input_image_bytes = get_png_encoders(tf.pack(inputs))

    saver = tf.train.Saver(tf.trainable_variables())

    sess = tf.Session()
    with sess.as_default():
        model_path = tf.train.latest_checkpoint(FLAGS.results_dir)
        print('..attempting to restore from {}'.format(model_path))
        saver.restore(sess, model_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        input_images, output_images = sess.run(
            [input_image_bytes, output_image_bytes])

        write_images(input_images, os.path.join(pic_dir, 'inputs'))
        write_images(output_images, os.path.join(pic_dir, 'states'))
>>>>>>> 542e34756b2971ff47dde981c264a1545bfd4f2c
