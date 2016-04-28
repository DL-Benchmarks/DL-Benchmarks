# Copyright (c) 2016 Robert Bosch LLC, USA.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This source code is based on Theano tutorial
#   http://deeplearning.net/tutorial/
# licensed under a BSD license found in the
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

"""Timing of LeNet."""

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
import numpy as np
import theano
import theano.tensor as T
from layers import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, relu

class Config(object):
    """params"""
    image_width = 28
    ydim = 10
    batch_size = 64
    num_kerns = [20, 50]  # number of kernels for first and second conv layers
    num_timing_iters = 200
    num_warmup_iters = 200


def build_lenet(config):
    rng = np.random.RandomState(23455)

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector

    image_width = config.image_width
    batch_size = config.batch_size
    image_size = image_width**2

    x_shared = T.cast(theano.shared(np.random.rand(batch_size, image_size),
                                    borrow=True), theano.config.floatX)
    y_shared = T.cast(theano.shared(np.random.randint(config.ydim,
                                                      size=batch_size),
                                    borrow=True), 'int32')

    layer0_input = x.reshape((batch_size, 1, image_width, image_width))

    # construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, image_width, image_width),
        filter_shape=(config.num_kerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, config.num_kerns[0], 12, 12),
        filter_shape=(config.num_kerns[1], config.num_kerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=config.num_kerns[1] * 4 * 4,
        n_out=500,
        activation=relu
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500,
                                n_out=config.ydim)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params_W = [layer3.W, layer2.W, layer1.W, layer0.W]
    params_b = [layer3.b, layer2.b, layer1.b, layer0.b]
    params = params_W + params_b

    shared_cost = theano.shared(np.float32(0.0))
    grads_temp = T.grad(cost, params)
    start_compilation = time.time()
    forward_step = theano.function([], [], updates=[(shared_cost, cost)],
                                   givens={x: x_shared, y: y_shared})
    forward_backward_step = theano.function([], grads_temp,
                                            givens={x: x_shared, y: y_shared})
    print 'compilation time: %.4f s' % (time.time() - start_compilation)
    return forward_step, forward_backward_step


def time_lenet():
    config = Config()

    forward_step, forward_backward_step = build_lenet(config)

    count = 0
    forward_time = np.zeros(config.num_timing_iters)
    forward_backward_time = np.zeros(config.num_timing_iters)
    num_iterations = config.num_warmup_iters + config.num_timing_iters
    for i in xrange(num_iterations):
        if i >= config.num_warmup_iters:
            if theano.config.device == 'cpu':
                s = time.time()*1000  # in milliseconds
                forward_step()
                temp1 = time.time()*1000
                forward_time[count] = temp1 - s
                s = time.time()*1000
                forward_backward_step()
                temp2 = time.time()*1000
                forward_backward_time[count] = temp2 - s
            else:
                theano.sandbox.cuda.synchronize()
                s = time.time()*1000
                forward_step()
                theano.sandbox.cuda.synchronize()
                temp1 = time.time()*1000
                forward_time[count] = temp1 - s
                theano.sandbox.cuda.synchronize()
                s = time.time()*1000
                forward_backward_step()
                theano.sandbox.cuda.synchronize()
                temp2 = time.time()*1000
                forward_backward_time[count] = temp2 - s
            count += 1
        else:   # dry runs
            forward_step()
            forward_backward_step()

    print("time forward: %.4f +- %.4f ms, batch size: %d" %
          (np.mean(forward_time), np.std(forward_time), config.batch_size))
    print("time gradient: %.4f +- %.4f ms, batch size: %d" %
          (np.mean(forward_backward_time), np.std(forward_backward_time),
           config.batch_size))

if __name__ == '__main__':	
    time_lenet()
