# This source code is from theano_alexnet repository at
# https://github.com/uoguelph-mlrg/theano_alexnet
# Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
# This source code is licensed under a BSD license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

# The original code was modified by Robert Bosch LLC, USA for timing.
# All modifications are licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------------


import theano
import theano.tensor as T
import time
import numpy as np
from layers import ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config.batch_size
        lib_conv = config.lib_conv
        group = (2 if config.grouping else 1)
        LRN = (True if config.LRN else False)
        print 'LRN, group', LRN, group

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        x = T.ftensor4('x')
        y = T.lvector('y')


        print '... building the model with ConvLib %s, LRN %s, grouping %i ' \
              % (lib_conv, LRN, group)
        self.layers = []
        params = []
        weight_types = []

        layer1_input = x

        convpool_layer1 = ConvPoolLayer(
            input=layer1_input,
            image_shape=((3, 224, 224, batch_size) if lib_conv == 'cudaconvnet'
                         else (batch_size, 3, 227, 227)),
            filter_shape=((3, 11, 11, 96) if lib_conv == 'cudaconvnet'
                          else (96, 3, 11, 11)),
            convstride=4,
            padsize=(0 if lib_conv == 'cudaconvnet' else 3),
            group=1,
            poolsize=3, poolstride=2,
            bias_init=0.0, lrn=LRN,
            lib_conv=lib_conv)
        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = ConvPoolLayer(
            input=convpool_layer1.output,
            image_shape=((96, 27, 27, batch_size) if lib_conv == 'cudaconvnet'
                         else (batch_size, 96, 27, 27)),
            filter_shape=((96, 5, 5, 256) if lib_conv == 'cudaconvnet'
                          else (256, 96, 5, 5)),
            convstride=1, padsize=2, group=group,
            poolsize=3, poolstride=2,
            bias_init=0.1, lrn=LRN,
            lib_conv=lib_conv,
            )
        self.layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type

        convpool_layer3 = ConvPoolLayer(
            input=convpool_layer2.output,
            image_shape=((256, 13, 13, batch_size) if lib_conv == 'cudaconvnet'
                         else (batch_size, 256, 13, 13)),
            filter_shape=((256, 3, 3, 384) if lib_conv == 'cudaconvnet'
                          else (384, 256, 3, 3)),
            convstride=1, padsize=1, group=1,
            poolsize=1, poolstride=0,
            bias_init=0.0, lrn=False,
            lib_conv=lib_conv,
            )
        self.layers.append(convpool_layer3)
        params += convpool_layer3.params
        weight_types += convpool_layer3.weight_type

        convpool_layer4 = ConvPoolLayer(
            input=convpool_layer3.output,
            image_shape=((384, 13, 13, batch_size) if lib_conv == 'cudaconvnet'
                         else (batch_size, 384, 13, 13)),
            filter_shape=((384, 3, 3, 384) if lib_conv == 'cudaconvnet'
                          else (384, 384, 3, 3)),
            convstride=1, padsize=1, group=group,
            poolsize=1, poolstride=0,
            bias_init=0.1, lrn=False,
            lib_conv=lib_conv,
            )
        self.layers.append(convpool_layer4)
        params += convpool_layer4.params
        weight_types += convpool_layer4.weight_type

        convpool_layer5 = ConvPoolLayer(
            input=convpool_layer4.output,
            image_shape=((384, 13, 13, batch_size) if lib_conv == 'cudaconvnet'
                         else (batch_size, 384, 13, 13)),
            filter_shape=((384, 3, 3, 256) if lib_conv == 'cudaconvnet'
                          else (256, 384, 3, 3)),
            convstride=1, padsize=1, group=group,
            poolsize=3, poolstride=2,
            bias_init=0.0, lrn=False,
            lib_conv=lib_conv,
            )
        self.layers.append(convpool_layer5)
        params += convpool_layer5.params
        weight_types += convpool_layer5.weight_type

        if lib_conv == 'cudaconvnet':
            fc_layer6_input = T.flatten(
               convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
        else:
            fc_layer6_input = convpool_layer5.output.flatten(2)

        fc_layer6 = FCLayer(input=fc_layer6_input, n_in=9216, n_out=4096)
        self.layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(fc_layer6.output)

        fc_layer7 = FCLayer(input=dropout_layer6.output, n_in=4096, n_out=4096)
        self.layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        dropout_layer7 = DropoutLayer(fc_layer7.output)

        softmax_layer8 = SoftmaxLayer(
            input=dropout_layer7.output, n_in=4096, n_out=1000)
        self.layers.append(softmax_layer8)
        params += softmax_layer8.params
        weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################

        self.cost = softmax_layer8.negative_log_likelihood(y)
        self.errors = softmax_layer8.errors(y)
        self.errors_top_5 = softmax_layer8.errors_top_x(y, 5)
        self.params = params
        self.x = x
        self.y = y
        # self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size


def compile_models(model, config, flag_top_5=False):

    x = model.x
    y = model.y

    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5
    batch_size = model.batch_size

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    if config.lib_conv == 'cudaconvnet':
        raw_size = 224
    else:
        raw_size = 227

    shared_x = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype=int),
                             borrow=True)

    # Define Theano Functions for timings
    shared_cost = theano.shared(np.float32(0.0))
    start_compilation = time.time()
    forward_step = theano.function(
        [], [],
        updates=[(shared_cost, cost)],
        givens=[(x, shared_x), (y, shared_y)])

    # compute the gradient using input which requires forward and backward steps
    # internally to calculate activations etc...
    forward_backward_step = theano.function(
        [], grads,
        givens=[(x, shared_x), (y, shared_y)])

    print 'compilation time %.4f s' % (time.time() - start_compilation)

    validate_outputs = [cost, errors]
    if flag_top_5:
        validate_outputs.append(errors_top_5)

    return forward_step, forward_backward_step, shared_x, shared_y
