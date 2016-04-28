# This source code is from Theano tutorial
#    http://deeplearning.net/tutorial/
# This source code is licensed under a BSD license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

# The original code was modified by Robert Bosch LLC, USA for timing.
# All modifications are licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Timing of Stacked AutoEncoder in theano"""

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox import cuda
import time
from layers import LogisticRegression, HiddenLayer


class Config(object):
    """params"""
    hidden_layers_sizes = [400, 200, 100]  # bigger version 800, 1000, 2000
    num_warmup_iters = 400      # number of dry runs before timing
    num_timing_iters = 800      # number of iterations to average over
    ydim = 10
    batch_size = 64
    image_size = 28 ** 2            # number pixels input


class StackedAutoEncoder(object):
    """Stacked auto-encoder class (SAE)
    Adopted from:
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/SdA.py

    A stacked autoencoder (SAE) model is obtained by stacking several
    AEs. The hidden layer of the AE at layer `i` becomes the input of
    the AE at layer `i+1`. The first layer AE gets as input the input of
    the SAE, and the hidden layer of the last AE represents the output.
    Note that after pretraining, the SAE is dealt with as a normal MLP,
    the AEs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        train_set_x,
        train_set_y,
        hidden_layers_sizes,
        n_ins=784,
        n_outs=10
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: np.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type train_set_x: theano.shared float32
        :param: train_set_x: Training data set, shape (n_samples, n_pixels)

        :type train_set_y: theano.shared, int32
        :param: train_set_x: GT for training data, shape (n_samples)

        :type n_ins: int
        :param n_ins: dimension of the input to the SAE

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
               at least one value
        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.AE_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

        assert self.n_layers > 0

        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of

        for i in xrange(self.n_layers):     # used to be n layers

            # construct the sigmoid layer = encoder stack
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=(n_ins if i == 0 else
                                              hidden_layers_sizes[i-1]),
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # init the DA_layer, takes weights from sigmoid layer
            AE_layer = AutoEncoder(
                numpy_rng=numpy_rng,
                input=layer_input,
                n_visible=(n_ins if i == 0 else hidden_layers_sizes[i-1]),
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.W,
                bhid=sigmoid_layer.b)

            self.AE_layers.append(AE_layer)

        # on top of the layers
        # log layer for fine-tuning
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, batch_size):
        """
        Generates a list of functions to time each AE training.

        :type batch_size: int
        :param batch_size: size of a [mini]batch
        """

        index = T.lscalar('index')  # index to a minibatch

        # beginning of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        forward_backward_step = []
        forward_step_fns = []
        i = 0
        for AE in self.AE_layers:

            # get the cost and the updates list
            cost = AE.get_cost_updates()

            params = AE.params
            shared_cost = theano.shared(np.float32(0.0))
            forward_step_fns.append(
                theano.function(
                    [index], [],
                    updates=[(shared_cost, cost)],
                    givens={
                            self.x: self.train_set_x[batch_begin: batch_end],
                            }))
            grads_temp = T.grad(cost, params)

            # This is both forward and backward
            forward_backward_step.append(
                theano.function(
                    [index], grads_temp,
                    givens={
                            self.x: self.train_set_x[batch_begin: batch_end],
                            }))
            i += 1

        return forward_backward_step, forward_step_fns

    def build_finetune_functions(self, batch_size):

        index = T.lscalar('index')  # index to a [mini]batch
        # beginning of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        cost = self.finetune_cost
        shared_cost = theano.shared(np.float32(0.0))
        forward_mlp = theano.function(
            [index], [],
            updates=[(shared_cost, cost)],
            givens={
                    self.x: self.train_set_x[batch_begin: batch_end],
                    self.y: self.train_set_y[batch_begin: batch_end],
                    })

        grads_temp = T.grad(cost, self.params)

        # This is both forward and backward
        forward_backward_mlp = theano.function(
            [index], grads_temp,
            givens={
                    self.x: self.train_set_x[batch_begin: batch_end],
                    self.y: self.train_set_y[batch_begin: batch_end],
                    })

        return forward_mlp, forward_backward_mlp


class AutoEncoder(object):
    """Denoising Auto-Encoder class (dA)
    Adopted from:
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/SdA.py

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details.
    """

    def __init__(
        self,
        numpy_rng,
        input,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
    ):
        """
        :type numpy_rng: np.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        self.params = [self.W, self.b, self.b_prime]

        self.hidden_values = None
        self.x = input

    def get_hidden_values(self):
        """ Computes the values of the hidden layer """

        return T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        y = self.get_hidden_values()
        z = self.get_reconstructed_input(y)

        L = T.sum((self.x-z)**2, axis=1)

        cost = T.mean(L)

        return cost

    def get_updates(self, learning_rate, cost):
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return updates


def time_theano_fn(fn, index, GPU_bool):
    if GPU_bool:
        theano.sandbox.cuda.synchronize()
    start = time.time()*1000
    fn(index)
    if GPU_bool:
        theano.sandbox.cuda.synchronize()
    elapsed_time = time.time()*1000 - start
    return elapsed_time


def time_SAE():
    """
    Time autoencoder. Time training of each AE layer and the finetuning of the
    encoder stack.
    """
    config = Config()
    hidden_layers_sizes = config.hidden_layers_sizes
    n_warmup_iter = config.num_warmup_iters
    n_timing_iter = config.num_timing_iters
    batch_size = config.batch_size
    n_ins = config.image_size
    n_out = config.ydim
    if theano.config.device == 'gpu':
        gpu_bool = True
        print 'GPU: synchronizing before and after timing'
    else:
        gpu_bool = False

    x = theano.shared(np.random.random((batch_size, n_ins)).astype(
        dtype=theano.config.floatX), borrow=True)
    y = T.cast(theano.shared(np.random.randint(n_out, size=batch_size),
                             borrow=True), 'int32')

    print '... building the model'
    numpy_rng = np.random.RandomState(89677)
    SAE = StackedAutoEncoder(
              numpy_rng=numpy_rng,
              n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes,
              train_set_x=x,
              train_set_y=y,
    )

    print '... compiling pretraining and finetuning functions'
    start_compilation = time.time()
    forward_backward_step, forward_step_fns = \
        SAE.pretraining_functions(batch_size=batch_size)

    forward_mlp, forward_backward_mlp = \
        SAE.build_finetune_functions(batch_size=batch_size)
    print 'compilation time %.4f s' % (time.time() - start_compilation)

    print 'Timing the model'
    t_forwards_ae = np.zeros((len(hidden_layers_sizes), n_timing_iter))
    t_forw_back_ae = np.zeros((len(hidden_layers_sizes), n_timing_iter))
    t_forward_mlp = np.zeros(n_timing_iter)
    t_forw_back_mlp = np.zeros(n_timing_iter)

    for i in xrange(SAE.n_layers):
        # go through pretraining epochs
        iterations = 0

        for counter in range(n_warmup_iter + n_timing_iter):
            # dry runs
            if iterations < n_warmup_iter:
                forward_step_fns[i](0)
                forward_backward_step[i](0)

            # timing (over-specified below for clearness)
            if n_warmup_iter <= iterations < n_warmup_iter + n_timing_iter:
                t_forwards_ae[i, iterations-n_warmup_iter] = \
                    time_theano_fn(forward_step_fns[i], 0, gpu_bool)

                t_forw_back_ae[i, iterations-n_warmup_iter] = \
                    time_theano_fn(forward_backward_step[i], 0, gpu_bool)

            iterations += 1

    # update steps
    iterations = 0
    for counter in range(n_warmup_iter + n_timing_iter):
        if n_warmup_iter <= iterations < n_warmup_iter + n_timing_iter:
            t_forward_mlp[iterations-n_warmup_iter] = \
                time_theano_fn(forward_mlp, 0, gpu_bool)

            t_forw_back_mlp[iterations-n_warmup_iter] = \
                time_theano_fn(forward_backward_mlp, 0, gpu_bool)

        else:
            forward_mlp(0)
            forward_backward_mlp(0)

        iterations += 1

    if gpu_bool:
        print '\n device GPU'
    print 'n_timing_iter %i n_dry %i batch size %i' \
          % (n_timing_iter, n_warmup_iter, batch_size)
    print 'layer sizes: in 784, hidden %s, out %i ' \
          % (tuple(hidden_layers_sizes), 10)
    for i in range(len(hidden_layers_sizes)):
        if np.any(t_forwards_ae <= 0.01) or np.any(t_forw_back_ae <= 0.01):
            print 'warining measurements problems'

        print 'AE %i forward: % .4f +- %.4f ms, batch size: %d' \
              % (i, np.mean(t_forwards_ae[i]), np.std(t_forwards_ae[i]),
                 batch_size)
        print 'AE %i forward + backward: % .4f +- %.4f ms, batch size: %d'\
              % (i, np.mean(t_forw_back_ae[i]), np.std(t_forw_back_ae[i]),
                 batch_size)

    print 'MLP forward: % .4f +- %.4f, batch size: %d' \
          % (np.mean(t_forward_mlp),  np.std(t_forward_mlp),  batch_size)
    print 'MLP forward + backward: % .4f +- %.4f ms, batch size: %d' \
          % (np.mean(t_forw_back_mlp), np.std(t_forw_back_mlp), batch_size)


if __name__ == '__main__':
    time_SAE()
