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

"""Timing of AlexNet."""

from neon.backends import gen_backend
from neon.data import ArrayIterator
from neon.initializers import Gaussian, GlorotUniform, Constant
from neon.layers import GeneralizedCost, Affine, Conv, Dropout, Pooling
from neon.models import Model
from neon.transforms import Rectlin, CrossEntropyMulti, Softmax
import numpy as np
import time
import pycuda.driver as drv


class Config(object):
    image_width = 224
    ydim = 1000
    batch_size = 256
    rng_seed = 23455
    backend = 'gpu'  # cpu or gpu
    num_warmup_iters = 200.0
    num_timing_iters = 200.0

config = Config()
# setup backend
be = gen_backend(backend=config.backend,
                 batch_size=config.batch_size,
                 rng_seed=config.rng_seed,
                 datatype=np.float32,
                 stochastic_round=False)

# generate data
X = np.random.rand(config.batch_size, config.image_width**2 * 3)
y = np.random.randint(config.ydim, size=config.batch_size)

# setup a training set iterator
data = ArrayIterator(X, y, nclass=config.ydim, lshape=(3, config.image_width,
                                                       config.image_width))

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)
init1 = Gaussian(scale=0.01)
init2 = Gaussian(scale=0.03)
init_uni = GlorotUniform()

# setup model layers
# no local response normalization
relu = Rectlin()
layers = []
layers.append(Conv((11, 11, 96), strides=4, padding=0, init=init1,
                   bias=Constant(0), activation=relu))
layers.append(Pooling(3, strides=2))
layers.append(Conv((5, 5, 256), padding=2, init=init1, bias=Constant(1),
                   activation=relu))
layers.append(Pooling(3, strides=2))
layers.append(Conv((3, 3, 384), padding=1, init=init2, bias=Constant(0),
                   activation=relu))
layers.append(Conv((3, 3, 384), padding=1, init=init2, bias=Constant(1),
                   activation=relu))
layers.append(Conv((3, 3, 256), padding=1, init=init2, bias=Constant(1),
                   activation=relu))
layers.append(Pooling(3, strides=2))
layers.append(Affine(nout=4096, init=init1, bias=Constant(1), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=4096, init=init1, bias=Constant(1), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=1000, init=init1, bias=Constant(-7),
                     activation=Softmax()))

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# initialize model object
network = Model(layers=layers)
network.cost = cost
network.initialize(data, cost)

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
        if iter > config.num_warmup_iters:  # time it
            if config.backend == 'cpu':
                s = time.time()*1000
                x = network.fprop(x)
                cost_iter = network.cost.get_cost(x, t)
                e = time.time()*1000  # in milliseconds
                forward_time[iter - config.num_warmup_iters - 1] = e - s
                s = time.time()*1000
                delta = network.cost.get_errors(x, t)  # gradient of the cost
                network.bprop(delta)
                e = time.time()*1000
                backward_time[iter - config.num_warmup_iters - 1] = e - s
            else:
                start.record()
                x = network.fprop(x)
                cost_iter = network.cost.get_cost(x, t)
                end.record()
                end.synchronize()
                forward_time[iter - config.num_warmup_iters - 1] \
                    = end.time_since(start)
                start.record()
                delta = network.cost.get_errors(x, t)
                network.bprop(delta)
                end.record()
                end.synchronize()
                backward_time[iter - config.num_warmup_iters - 1] \
                    = end.time_since(start)
        else: # warmups
            x = network.fprop(x)
            delta = network.cost.get_errors(x, t)
            network.bprop(delta)

print("time forward: %.4f +- %.4f ms, batch size: %d" %
      (np.mean(forward_time), np.std(forward_time), config.batch_size))
print("time gradient: %.4f +- %.4f ms, batch size: %d" %
      (np.mean(forward_time + backward_time),
       np.std(forward_time+ backward_time), config.batch_size))
