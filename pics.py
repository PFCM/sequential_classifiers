"""Make some pretty pictures of the networks outputs/hidden state while running over a 
test image"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import mrnn
import rnndatasets.sequentialmnist as data

import sequential_model as sm
import mnist  # also gives us the flags we need


flags = tf.app.flags
flags.DEFINE_string('model_path', None, 'Where to look for a model to load.')

FLAGS = flags.FLAGS


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


# need some way of associating them with the correct image
def write_images(images, name):
    """Gets a numpy array of appropriate bytes and writes each to a file"""
    print(images.shape)
    for i, image in enumerate(images):
        img_name = os.path.join(FLAGS.results_dir, name + '{}.png'.format(i))
        with open(img_name, 'wb') as fp:
            fp.write(bytes(image))


if __name__ == '__main__':
    # serious unrolling
    seq_length = 28*28

    inputs = [tf.placeholder(tf.float32, name='input_{}'.format(i),
                             shape=[FLAGS.batch_size, 1])
              for i in range(seq_length)]

    # get the model
    print('...getting model')
    with tf.variable_scope('model'):
        cell = mnist.get_cell(FLAGS.width)
        init_state, final_state, logits, outputs = sm.inference(
            inputs, FLAGS.layers, cell, 10, full_logits=True)
    print('got model')

    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)

    output_image_bytes = get_png_encoders(tf.pack(outputs))
    input_image_bytes = get_png_encoders(tf.pack(inputs))
    # somehow this seems like wasted effort, but anyway
    # make the images into a [width x batch x height]
    # if we pack them they are [width*height x batch x 1]
    imgs = tf.transpose(tf.squeeze(tf.pack(inputs)))
    # now it is [batch x width*height]
    imgs = tf.reshape(imgs, [-1, 28, 28])
    imgs = tf.transpose(imgs, [2, 0, 1])
    square_image_bytes = get_png_encoders(imgs)
    classifier_image_bytes = get_png_encoders(tf.pack([tf.nn.softmax(logit) for logit in logits]))

    saver = tf.train.Saver(tf.trainable_variables())

    if FLAGS.permute:
        permutation = data.get_permutation(1001)
    else:
        permutation = None
    __, _, test = data.get_iters(FLAGS.batch_size, shuffle=False, permute=permutation)

    sess = tf.Session()

    with sess.as_default():
        restore_path = FLAGS.model_path
        if os.path.isdir(restore_path):
            restore_path = tf.train.latest_checkpoint(restore_path)
            if not restore_path:
                raise ValueError('Could not find model in `{}`'.format(FLAGS.model_path))
        print('...attempting to restore from {}'.format(restore_path))
        saver.restore(sess, restore_path)
        print('...done')

        # run a batch
        batch = next(test)
        input_images, output_images, square_images, class_images = sess.run(
            [input_image_bytes, output_image_bytes, square_image_bytes,
             classifier_image_bytes],
            {inputs[i]: batch[0][i, ...] for i in range(len(inputs))})

        # and write them
        write_images(input_images, 'input')
        write_images(output_images, 'output')
        write_images(square_images, 'digit')
        write_images(class_images, 'class')
