# This source code is from Theano tutorial
#    http://deeplearning.net/tutorial/lstm.html
# This source code is licensed under a BSD license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

# The original code was modified by Robert Bosch LLC, USA for timing.
# All modifications are licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Timing of LSTM.
LSTM network is implemented on Theano for binary classification. Masking
is used to handle variable length inputs. The network consists of an
embedding layer, followed by lstm layer and finally a classifier. The lstm
layer averages the outputs for each sample of the sequence to find an
aggregated feature of the sequence which is then fed to the classifier.
"""

from collections import OrderedDict
import time
import numpy as np
import theano
from theano import config as T_config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class Config(object):
    """## params"""
    n_words = 10000
    maxlen = 100  # the number of unrolled steps of LSTM during training
    batch_size = 16
    vocab_size = 10000
    dim_proj = 128  # word embeding dimension and number of hidden units
    num_timing_epochs = 5
    num_warmup_epochs = 1
    ydim = 2
    use_dropout = True

SEED = 123
np.random.seed(SEED)


def numpy_floatX(data):
    return np.asarray(data, dtype=T_config.floatX)


def get_minibatches_idx(n, batch_size, shuffle=False):
    """
    Divide samples into batches
    n : number of samples
    batch_size
    shuffle: turn on shuffle at each iteration.

    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def prepare_data(seqs, labels, maxlen=None):
    """Create the inputs and mask from the dataset.

    This pads each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape, p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(config):
    """
    parameters for different layers including embedding, lstm, and  classifier.
    """
    params = OrderedDict()
    # embedding layer parameter
    randn = np.random.rand(config.n_words, config.dim_proj)
    params['Wemb'] = (0.01 * randn).astype(T_config.floatX)
    
    # lstm layer parameters
    params = param_init_lstm(config, params, prefix='lstm')

    # classifier parameters
    params['U'] = 0.01 * np.random.randn(config.dim_proj,
                                         config.ydim).astype(T_config.floatX)
    params['b'] = np.zeros((config.ydim,)).astype(T_config.floatX)

    return params


def init_tparams(params):
    # making shared variables and initializing them
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(T_config.floatX)

    
def param_init_lstm(config, params, prefix='lstm'):
    """
    Init the LSTM parameter and attach it to the exitings params

    :see: init_params
    """
    # each LSTM cell has 4 weight matrices for input and 4 weight matrices for
    # state
    W = np.concatenate([ortho_weight(config.dim_proj)]*4, axis=1)
    params[_p(prefix, 'W')] = W
    U = np.concatenate([ortho_weight(config.dim_proj)]*4, axis=1)
    params[_p(prefix, 'U')] = U
    b = np.zeros((4 * config.dim_proj,))
    params[_p(prefix, 'b')] = b.astype(T_config.floatX)

    return params


def lstm_layer(tparams, state_below, config, prefix='lstm', mask=None):
    """ state_below is a n_timesteps*n_samples(batchSize)*embedding_size where
    n_timesteps is the length of the longest sentense in current batch.
    tparams contains the lstm parameters.
    The matrices are concatinated into one tensor for parallelizing the
    computations of inputs to different gates.
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1  # mini batch size is 1.

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, config.dim_proj))
        f = tensor.nnet.sigmoid(_slice(preact, 1, config.dim_proj))
        o = tensor.nnet.sigmoid(_slice(preact, 2, config.dim_proj))
        c = tensor.tanh(_slice(preact, 3, config.dim_proj))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           config.dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           config.dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]  # Note that rval is 2d (output and state).


def build_model(tparams, config):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=T_config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]  # length of longest sequence in batch
    n_samples = x.shape[1]  # batch size

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                config.dim_proj])
    proj = lstm_layer(tparams, emb, config, prefix='lstm', mask=mask)

    # find the averaged representation for each sentence
    proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]
    if config.use_dropout:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, cost


def time_lstm():
    config = Config()

    # The data is similar to IMDB review train data with maximum sequence
    # length of 100 which can be found at
    # http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl
    print 'Prepare data'
    seqs_lengths = np.genfromtxt("../../data/seqs_lengths.csv")
    num_samples = len(seqs_lengths)
    seqs = [np.random.randint(1, config.n_words, item) for item in seqs_lengths]
    labels = np.random.randint(config.ydim, size=num_samples)
    data = [seqs, labels]

    print 'Building model'
    params = init_params(config)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, cost) = build_model(tparams, config)

    compile_start = time.time()

    grads = tensor.grad(cost, wrt=tparams.values())

    # Define Theano functions for timings
    shared_cost = theano.shared(numpy_floatX(0.0))
    forward_step = theano.function([x, mask, y], [],
                                   updates=[(shared_cost, cost)])
    forward_backward_step = theano.function([x, mask, y], grads)

    print('compile time: %.4f' % (time.time() - compile_start))
    print 'Optimization'

    # do not shuffle so that the batches used for timing across different runs
    # and platforms have the same dimensionality. Note that each batch have
    # the length of its maximum length sequence.
    kf = get_minibatches_idx(len(data[0]), config.batch_size, shuffle=False)

    forward_time = np.zeros(config.num_timing_epochs)
    forward_backward_time = np.zeros(config.num_timing_epochs)
    num_epochs = config.num_warmup_epochs + config.num_timing_epochs
    for eidx in xrange(num_epochs):
        if eidx >= config.num_warmup_epochs:
            forward_time_epoch = 0
            forward_backward_time_epoch = 0
            num_iter = 0.0
            if theano.config.device == 'cpu':
                for _, train_index_timing in kf:
                    num_iter += 1.0
                    y = [data[1][t] for t in train_index_timing]
                    x = [data[0][t]for t in train_index_timing]
                    x, mask, y = prepare_data(x, y)
                    s = time.time()*1000
                    forward_step(x, mask, y)
                    temp1 = time.time()*1000
                    forward_time_epoch += temp1 - s
                    forward_backward_step(x, mask, y)
                    temp2 = time.time()*1000
                    forward_backward_time_epoch += temp2 - temp1
            else:
                for _, train_index_timing in kf:
                    num_iter += 1.0
                    y = [data[1][t] for t in train_index_timing]
                    x = [data[0][t]for t in train_index_timing]
                    x, mask, y = prepare_data(x, y)
                    theano.sandbox.cuda.synchronize()
                    s = time.time()*1000
                    forward_step(x, mask, y)
                    theano.sandbox.cuda.synchronize()
                    temp1 = time.time()*1000
                    forward_time_epoch += temp1 - s
                    theano.sandbox.cuda.synchronize()
                    s = time.time()*1000
                    forward_backward_step(x, mask, y)
                    theano.sandbox.cuda.synchronize()
                    temp2 = time.time()*1000
                    forward_backward_time_epoch += temp2 - s
            forward_time[eidx-config.num_warmup_epochs] = \
                forward_time_epoch / num_iter
            forward_backward_time[eidx-config.num_warmup_epochs] = \
                forward_backward_time_epoch / num_iter

        else:  # warmup runs
            for _, train_index_timing in kf:
                y = [data[1][t] for t in train_index_timing]
                x = [data[0][t]for t in train_index_timing]
                x, mask, y = prepare_data(x, y)
                forward_step(x, mask, y)
                forward_backward_step(x, mask, y)

    print("time forward: %.4f +- %.4f ms, batch size: %d" %
          (np.mean(forward_time), np.std(forward_time), config.batch_size))
    print("time gradient: %.4f +- %.4f ms, batch size: %d" %
          (np.mean(forward_backward_time), np.std(forward_backward_time),
           config.batch_size))
    return


if __name__ == '__main__':
    time_lstm()
