# Copyright (c) 2016 Robert Bosch LLC, USA.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This source code is based on Soumith Chintala's benchmarking code
#    https://github.com/soumith/convnet-benchmarks
# Licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

"""Timing of stacked auto-encoders."""

import time
import tensorflow as tf
import numpy as np

parameters = []
affine_counter = 1

class Config(object):
    """params"""
    image_width = 28
    ydim = 10   # number of classes
    batch_size = 64
    num_timing_iters = 200
    num_warmup_iters = 200
    CPU_only = False  # selection between CPU and GPU
    num_threads = 12  # number of threads in the CPU only mode
    encoder_size = [400, 200, 100]  # number of hidden units for each encoder


def _linear(inpOp, nIn, nOut):
    global parameters
    name = 'linear'
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        lin = tf.matmul(inpOp, kernel) + biases
        parameters += [kernel, biases]
        return lin


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
        affine = tf.sigmoid(tf.matmul(inpOp, kernel) + biases, name=name)
        parameters += [kernel, biases]
        return affine


def _affineShared(inpOp, nIn, nOut):
    """affine layer with weight sharing and independent biases"""
    global affine_counter
    global parameters
    affine_counter -= 1  # for weight sharing
    name = 'affineDec' + str(affine_counter)
    name_kernel = 'affine' + str(affine_counter)
    
    with tf.name_scope(name) as scope:
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        parameters += [biases]
        
        with tf.variable_scope(name_kernel):
            kernel = tf.get_variable('weights', [nOut, nIn])
        
        affine = tf.sigmoid(tf.matmul(inpOp, tf.transpose(kernel)) + biases,
                             name=name)
        return affine


def loss_classifier(logits, labels, config):
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


def loss_reconstruction(images_recons, featureMaps, config):
    loss_recons = tf.reduce_sum(tf.square(images_recons - featureMaps)) / \
                  (2 * config.batch_size)
    
    return loss_recons


def inference_AE1(images, config):
    resh1 = tf.reshape(images, [-1, config.image_width**2])
    enc1 = _affine(resh1, config.image_width**2, config.encoder_size[0])
    dec1 = _affineShared(enc1, config.encoder_size[0], config.image_width**2)
    
    return [images, dec1]  # input and output of the auto-encoder layer


def inference_AE2(images, config):
    resh1 = tf.reshape(images, [-1, config.image_width**2])
    enc1 = _affine(resh1, config.image_width**2, config.encoder_size[0])
    enc2 = _affine(enc1, config.encoder_size[0], config.encoder_size[1])
    dec2 = _affineShared(enc2, config.encoder_size[1], config.encoder_size[0])
    
    return [enc1, dec2]  # input and output of the auto-encoder layer


def inference_AE3(images, config):
    resh1 = tf.reshape(images, [-1, config.image_width**2])
    enc1 = _affine(resh1, config.image_width**2, config.encoder_size[0])
    enc2 = _affine(enc1, config.encoder_size[0], config.encoder_size[1])
    enc3 = _affine(enc2, config.encoder_size[1], config.encoder_size[2])
    dec3 = _affineShared(enc3, config.encoder_size[2], config.encoder_size[1])

    return [enc2, dec3]  # input and output of the auto-encoder layer


def inference_finetuning(images, config):
    resh1 = tf.reshape(images, [-1, config.image_width**2])
    enc1 = _affine(resh1, config.image_width**2, config.encoder_size[0])
    enc2 = _affine(enc1, config.encoder_size[0], config.encoder_size[1])
    enc3 = _affine(enc2, config.encoder_size[1], config.encoder_size[2])
    classifier_outputs = _linear(enc3, config.encoder_size[2], config.ydim)
    
    return classifier_outputs


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


def run_benchmark():
    global parameters
    global affine_counter
    config = Config()
    with tf.Graph().as_default():
        images = tf.Variable(tf.random_normal([config.batch_size,
                                               config.image_width**2],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        encoder_decoder_outs = inference_AE1(images, config)

        inter_layer1 = encoder_decoder_outs[0]
        last_layer1 = encoder_decoder_outs[1]

        # Build an initialization operation.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        if config.CPU_only:
            device_config = tf.ConfigProto(
                device_count={'GPU': 0},
                intra_op_parallelism_threads=config.num_threads
            )
        else:
            device_config = tf.ConfigProto(device_count = {'GPU': 1})
        print device_config
        sess = tf.Session(config=device_config)
        sess.run(init)

        # Run the forward benchmark.
        time_tensorflow_run(sess, last_layer1, "Forward", config)

        # Add a simple objective so we can calculate the backward pass.
        objective1 = loss_reconstruction(last_layer1, inter_layer1, config)

        # Compute gradients with respect to parameters of 1st AE.
        print("Parameters of 1st AE are:")
        print([param.name for param in parameters])
        grad1 = tf.gradients(objective1, parameters)

        # Run the backward benchmark.
        time_tensorflow_run(sess, grad1, "Forward-backward", config)

        print("Done with AE1 !!!!!!!!!!!!!!!!! \n")
    
        # Run benchmark for AE2
        parameters = []
        affine_counter = 1

        encoder_decoder_outs = inference_AE2(images, config)

        inter_layer2 = encoder_decoder_outs[0]
        last_layer2 = encoder_decoder_outs[1]
    
        # Build an initialization operation.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Run the forward benchmark.
        time_tensorflow_run(sess, last_layer2, "Forward", config)

        objective2 = loss_reconstruction(last_layer2, inter_layer2, config)

        # Compute gradients with respect to parameters of 2nd AE.
        AE2_params = parameters[2:]
        print("Parameters of 2nd AE are:")
        print([param.name for param in AE2_params])
        grad2 = tf.gradients(objective2, AE2_params)

        time_tensorflow_run(sess, grad2, "Forward-backward", config)

        print("Done with AE2 !!!!!!!!!!!!!!!!! \n")

        # Run benchmark on AE3
        parameters = []
        affine_counter = 1
        encoder_decoder_outs = inference_AE3(images, config)

        inter_layer3 = encoder_decoder_outs[0]
        last_layer3 = encoder_decoder_outs[1]

        # Build an initialization operation.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Run the forward benchmark.
        time_tensorflow_run(sess, last_layer3, "Forward", config)

        # Add a simple objective so we can calculate the backward pass.
        objective3 = loss_reconstruction(last_layer3, inter_layer3, config)

        # Compute gradients with respect to parameters of 3rd AE.
        AE3_params = parameters[4:]
        print("Parameters of 3rd AE are:")
        print([param.name for param in AE3_params])
        grad3 = tf.gradients(objective3, AE3_params)

        # Run the backward benchmark.
        time_tensorflow_run(sess, grad3, "Forward-backward", config)

        print("Done with AE3 !!!!!!!!!!!!!!!!! \n")

        # Run benchmark on Finetuning Step
        parameters = []
        affine_counter = 1

        # generate random labels for the fine-tuning step
        labels = tf.Variable(
            np.random.randint(config.ydim,
                              size=config.batch_size).astype(np.int32))

        last_layer4 = inference_finetuning(images, config)

        # Build an initialization operation.
        init = tf.initialize_all_variables()
        sess = tf.Session(config=device_config)
        sess.run(init)

        # Run the forward benchmark.
        time_tensorflow_run(sess, last_layer4, "Forward", config)

        # Add a simple objective so we can calculate the backward pass.
        objective4 = loss_classifier(last_layer4, labels, config)

        # Compute gradients with respect to parameters of all encoders (not
        # including decoders) and last fully connected layer
        print("Parameters of fine-tuning step are:")
        print([param.name for param in parameters])
        grad4 = tf.gradients(objective4, parameters)

        # Run the backward benchmark.
        time_tensorflow_run(sess, grad4, "Forward-backward", config)

        print("Done with Finetuning !!!!!!!!!!!!!!!!! \n")
    

def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
