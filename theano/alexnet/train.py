# Copyright (c) 2016 Robert Bosch LLC, USA.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------------

"""Timing of AlexNet."""

import numpy as np
import pycuda.driver as drv
import theano
import theano.sandbox.cuda
import theano.misc.pycuda_init
import theano.misc.pycuda_utils
from alex_net import AlexNet, compile_models

theano.sandbox.cuda.use('gpu0')
theano.config.on_unused_input = 'warn'


class Config(object):
    """params"""
    batch_size = 256
    grouping = True      # Turn on grouping of conv layers 2, 4, 5
    LRN = True           # Turn on LRN for conv layers 2,4,5
    lib_conv = 'cudnn'   # cudnn or cudaconvnet.
    num_timing_iters = 10
    num_warmup_iters = 20


def time_model(forward_step, forward_backward_step, shared_x, shared_y, config,
               batch_size):
    if config.lib_conv == 'cudaconvnet':
        # c01b
        batch_img = np.random.normal(0, 0.1, size=(3, 224, 224, batch_size))\
            .astype(np.float32)
    else:
        # bc01
        batch_img = np.random.normal(0, 0.1, size=(batch_size, 3, 227, 227))\
            .astype(np.float32)

    shared_x.set_value(batch_img)

    batch_label = np.random.randint(0, 1000, size=batch_size).astype(np.int64)
    shared_y.set_value(batch_label)

    start = drv.Event()
    end = drv.Event()

    start.record()
    start.synchronize()
    forward_step()
    end.record()
    end.synchronize()
    forward_time = end.time_since(start)

    start.record()
    start.synchronize()
    forward_backward_step()
    end.record()
    end.synchronize()
    forward_backward_time = end.time_since(start)

    return forward_time, forward_backward_time


def time_alexnet():
    config = Config()

    # Build network
    model = AlexNet(config)
    batch_size = model.batch_size

    # Compile forward and forward-backward functions
    (forward_step, forward_backward_step, shared_x, shared_y) = \
        compile_models(model, config)

    num_warmup_iters = config.num_warmup_iters
    num_timing_iters = config.num_timing_iters
    forward_times = np.zeros(num_timing_iters)
    forward_backward_times = np.zeros(num_timing_iters)
    num_iterations = num_warmup_iters + num_timing_iters
    for minibatch_index in range(num_iterations):
        if num_warmup_iters <= minibatch_index < num_warmup_iters + num_timing_iters:
            forward_time, forward_backward_time = \
                time_model(
                    forward_step, forward_backward_step, shared_x,
                    shared_y, config, batch_size)
            forward_times[minibatch_index-num_warmup_iters] = forward_time

            forward_backward_times[minibatch_index-num_warmup_iters] = \
                forward_backward_time

        else:   # Do not measure time for the warmup iterations.
            time_model(
                forward_step, forward_backward_step, shared_x,
                shared_y, config, batch_size)

    # print forward_times
    # print forward_backward_times
    print("forward time %.4f +- %.4f ms batch_size %d"
          % (np.mean(forward_times), np.std(forward_times), batch_size))
    print("gradient computation time %.4f +- %.4f ms batch_size %d"
          % (np.mean(forward_backward_times),  np.std(forward_backward_times),
             batch_size))


if __name__ == '__main__':
    time_alexnet()
