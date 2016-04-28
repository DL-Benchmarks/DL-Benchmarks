# This source code is from Soumith Chintala's benchmarking code
#   https://github.com/soumith/convnet-benchmarks
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

# The original code was modified by Robert Bosch LLC, USA to time AlexNet
# (adding dropout and local response normalization).
# All modifications are also licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Timing of AlexNet."""

import time
import tensorflow as tf
import numpy as np

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1


class Config(object):
    """params"""
    image_width = 224
    ydim = 1000
    batch_size = 256
    num_timing_iters = 200
    num_warmup_iters = 200
    CPU_only = False  # selection between CPU and GPU
    num_threads = 12  # number of threads in the CPU only mode
    SEED = 100  # Set to None for random seed.
    LRN = False  # use local response normalization or not
    gpu_memory_fraction = 0.32  # to measure memory consumption


def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1


def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1


def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    return tf.nn.max_pool(inpOp,
                          ksize=[1, kH, kW, 1],
                          strides=[1, dH, dW, 1],
                          padding='VALID',
                          name=name)


def loss(logits, labels, config):
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, config.batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([config.batch_size, config.ydim]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='entropy')
    loss = tf.reduce_mean(cross_entropy, name='entropy_mean')
    return loss


def inference(images, config):
    conv1 = _conv(images, 3, 96, 11, 11, 4, 4, 'VALID')
    pool1 = _mpool(conv1, 3, 3, 2, 2)
    if config.LRN:
        pool1 = tf.nn.local_response_normalization(pool1)
    conv2 = _conv(pool1, 96, 256, 5, 5, 1, 1, 'SAME')
    pool2 = _mpool(conv2, 3, 3, 2, 2)
    if config.LRN:
        pool2 = tf.nn.local_response_normalization(pool2)
    conv3 = _conv(pool2, 256, 384, 3, 3, 1, 1, 'SAME')
    conv4 = _conv(conv3, 384, 384, 3, 3, 1, 1, 'SAME')
    conv5 = _conv(conv4, 384, 256, 3, 3, 1, 1, 'SAME')
    pool5 = _mpool(conv5, 3, 3, 2, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    affn1 = _affine(resh1, 256 * 6 * 6, 4096)
    affn1 = tf.nn.dropout(affn1, 0.5, seed=config.SEED)
    affn2 = _affine(affn1, 4096, 4096)
    affn2 = tf.nn.dropout(affn2, 0.5, seed=config.SEED)
    affn3 = _affine(affn2, 4096, config.ydim)

    return affn3


def time_tensorflow_run(session, target, info_string, config):
    processing_times = np.zeros(config.num_timing_iters)
    if not isinstance(target, list):
        target = [target]
    for i in xrange(config.num_timing_iters + config.num_warmup_iters):
        start_time = time.time()
        _ = session.run(tf.group(*target))
        duration = time.time() - start_time
        if i >= config.num_warmup_iters:
            processing_times[i-config.num_warmup_iters] = duration
    print ('%s: %.4f +/- %.4f sec / batch, batch size: %d' %
           (info_string, np.mean(processing_times), np.std(processing_times),
            config.batch_size))


def time_alexnet():
    global parameters
    config = Config()
    with tf.Graph().as_default():
        # Note that our padding definition is slightly different the
        # cuda-convnet. In order to force the model to start with the same
        # activations sizes, we add 3 to the image_size and employ VALID
        # padding above.
        images = tf.Variable(tf.random_normal([config.batch_size,
                                               config.image_width + 3,
                                               config.image_width + 3, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        labels = tf.Variable(tf.ones([config.batch_size], dtype=tf.int32))

        # Build a Graph that computes the logits predictions from the
        # inference model.
        last_layer = inference(images, config)

        # Build an initialization operation.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        if config.CPU_only:
            device_config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=
                                        config.gpu_memory_fraction)
            device_config = tf.ConfigProto(gpu_options=gpu_options,
                                           device_count={'GPU': 1})
        print device_config
        sess = tf.Session(config=device_config)
        sess.run(init)

        # Run the forward benchmark.
        time_tensorflow_run(sess, last_layer, "Forward", config)

        # Add a simple objective so we can calculate the backward pass.
        objective = loss(last_layer, labels, config)

        # Compute the gradient with respect to all the parameters.
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward", config)


def main(_):
    time_alexnet()


if __name__ == '__main__':
    tf.app.run()
