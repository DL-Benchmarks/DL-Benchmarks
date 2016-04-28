# Copyright (c) 2016 Robert Bosch LLC, USA.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This source code is based on Neon
#     https://github.com/NervanaSystems/neon/
# Copyright 2015 Nervana Systems Inc., licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# ----------------------------------------------------------------------------

"""Timing of stacked auto-encoders."""

import sys
from neon.backends import gen_backend
from neon.data import ArrayIterator
from neon.initializers import Gaussian, GlorotUniform
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.transforms import Logistic, SumSquared, CrossEntropyMulti, Softmax
import numpy as np
import time
import pycuda.driver as drv


class Config(object):
    image_width = 28
    ydim = 10
    batch_size = 64
    rng_seed = 23455
    backend = 'gpu'  # cpu or gpu
    num_warmup_iters = 200
    num_timing_iters = 200
    encoder_size = [400, 200, 100]  # the size for each auto-encoder


def measure_time(data, network, config, network_name='unknown',
                    pre_training=True, create_target=False):
    """measure time for the auto-encoders and final mlp network.
    data is an iterator containing samples x and their label t.
    During pre-training, we may need to generate target t. This is controlled
    by create_target and pre_training inputs."""

    if config.backend == 'gpu':
        start = drv.Event()
        end = drv.Event()

    num_iterations = config.num_warmup_iters + config.num_timing_iters
    forward_time = np.zeros(config.num_timing_iters)
    backward_time = np.zeros(config.num_timing_iters)
    iter = 0
    flag = True
    while flag:
        for (x, t) in data:
            iter += 1
            if iter > num_iterations:
                flag = False
                break

            if pre_training:
                if create_target:
                    # helper network is used to create target output
                    len_network = len(network.layers.layers)
                    t = x  # target x
                    # last 4 layers are the actual encoder and decoder if the
                    # auto-encoder
                    if len_network > 4:
                        for i in range(len_network - 4):
                            # pass through the encoders only to get the target
                            t = network.layers.layers[i].fprop(t)
                    else:
                        sys.exit("something wrong with the configuration")
                else:
                    t = x

            if iter > config.num_warmup_iters:  # time it
                if config.backend == 'cpu':
                    s = time.time() * 1000
                    x = network.fprop(x)
                    network.cost.get_cost(x, t)
                    e = time.time() * 1000  # in milliseconds
                    forward_time[iter - config.num_warmup_iters - 1] = e - s
                    s = time.time() * 1000
                    delta = network.cost.get_errors(x, t)
                    network.bprop(delta)
                    e = time.time() * 1000
                    backward_time[iter - config.num_warmup_iters - 1] = e - s
                else:
                    start.synchronize()
                    start.record()
                    x = network.fprop(x)
                    network.cost.get_cost(x, t)
                    end.record()
                    end.synchronize()
                    forward_time[iter - config.num_warmup_iters - 1] \
                        = end.time_since(start)

                    start.synchronize()
                    start.record()
                    delta = network.cost.get_errors(x, t)
                    network.bprop(delta)
                    end.record()
                    end.synchronize()
                    backward_time[iter - config.num_warmup_iters - 1] \
                        = end.time_since(start)
            else:  # warm-up iterations
                x = network.fprop(x)
                delta = network.cost.get_errors(x, t)
                network.bprop(delta)

    print("time forward %s: %.4f +- %.4f ms, batch size: %d" %
          (network_name, np.mean(forward_time), np.std(forward_time),
           config.batch_size))
    print("time gradient %s: %.4f +- %.4f ms, batch size: %d" %
          (network_name, np.mean(forward_time + backward_time),
           np.std(forward_time + backward_time), config.batch_size))


config = Config()
image_size = config.image_width**2
# setup backendoptimizer_default
be = gen_backend(backend=config.backend,
                 batch_size=config.batch_size,
                 rng_seed=config.rng_seed,
                 datatype=np.float32,
                 stochastic_round=False)

# setup optimizer (no need to do this for timing)
# optimizer_default = GradientDescentMomentum(0.1, momentum_coef=1.0,
#                                             stochastic_round=False)
# optimizer_helper = GradientDescentMomentum(0.0, momentum_coef=1.0,
#                                           stochastic_round=False)

# generate data
X = np.random.rand(config.batch_size, config.image_width**2)
y = np.random.randint(config.ydim, size=config.batch_size)

# setup a training set iterator
data = ArrayIterator(X, y, nclass=config.ydim, lshape=(1, config.image_width,
                                                       config.image_width))

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)
init_uni = GlorotUniform()

# setting model layers for AE1
encoder1 = Affine(nout=config.encoder_size[0], init=init_norm,
                  activation=Logistic(), name='encoder1')
decoder1 = Affine(nout=image_size, init=init_norm, activation=Logistic(),
                  name='decoder1')
encoder2 = Affine(nout=config.encoder_size[1], init=init_norm,
                  activation=Logistic(), name='encoder2')
decoder2 = Affine(nout=config.encoder_size[0], init=init_norm,
                  activation=Logistic(), name='decoder2')
encoder3 = Affine(nout=config.encoder_size[2], init=init_norm,
                  activation=Logistic(), name='encoder3')
decoder3 = Affine(nout=config.encoder_size[1], init=init_norm,
                  activation=Logistic(), name='decoder3')
classifier = Affine(nout=config.ydim, init=init_norm, activation=Softmax())
cost_reconst = GeneralizedCost(costfunc=SumSquared()) 
cost_classification = GeneralizedCost(costfunc=CrossEntropyMulti())

# Setting model layers for AE1
AE1 = Model([encoder1, decoder1])
AE1.cost = cost_reconst
AE1.initialize(data, cost_reconst)
# AE1.optimizer = optimizer_default
measure_time(data, AE1, config, 'AE1')
            
# Setting model layers for AE2
# It has an extra encoder layer compared to what AE should really be. This is
# done to avoid saving the outputs for each AE.
AE2_mimic = Model([encoder1, encoder2, decoder2])
AE2_mimic.cost = cost_reconst
AE2_mimic.initialize(data, cost_reconst)
# Learning rates for extra layers that should not be updated are set to zero.
# opt = MultiOptimizer({'default': optimizer_default,
#                       'encoder1': optimizer_helper})
# AE2_mimic.optimizer = opt
measure_time(data, AE2_mimic, config, 'AE2', create_target=True)

# Setting model layers for AE3
AE3_mimic = Model([encoder1, encoder2, encoder3, decoder3])
AE3_mimic.cost = cost_reconst
AE3_mimic.initialize(data, cost_reconst)
# opt = MultiOptimizer({'default': optimizer_default,
#                       'encoder1': optimizer_helper,
#                       'encoder2': optimizer_helper})
# AE3_mimic.optimizer = opt
measure_time(data, AE3_mimic, config, 'AE3', create_target=True)

# Setting model layers for fine-tuning step
mlp = Model([encoder1, encoder2, encoder3, classifier])
mlp.cost = cost_classification
mlp.initialize(data, cost_classification)
# mlp.optimizer = optimizer_default
measure_time(data, mlp, config, 'mlp', pre_training=False)
