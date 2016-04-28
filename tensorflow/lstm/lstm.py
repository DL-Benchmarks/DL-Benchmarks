# Copyright (c) 2016 Robert Bosch LLC, USA.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#  This source code is based on TensorFlow project
#   https://github.com/tensorflow/tensorflow
# Copyright 2015 2015 Google Inc., licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# ==============================================================================

"""Timing of LSTM.
LSTM network is implemented on TensorFlow for binary classification. Masking
is used to handle variable length inputs. The network consists of an
embedding layer, followed by lstm layer and finally a classifier. The lstm
layer averages the outputs for each sample of the sequence to find an
aggregated feature of the sequence which is then fed to the classifier.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
import rnn_cell
import time


class Config(object):
  """## params"""
  n_words = 10000
  maxlen = 100  # the number of unrolled steps of LSTM during training
  batch_size = 16
  dim_proj = 128  # word embedding dimension and number of hidden units.
  keep_prob = 0.5  # if less than 1, it uses dropout on output of lstm layer
  num_timing_epochs = 5
  num_warmup_epochs = 1
  ydim = 2
  init_scale = 0.1
  CPU_only = False  # selection between CPU and GPU
  num_threads = 12  # number of threads in the CPU only mode

class LSTMNet(object):
  """The LSTM network for classification."""

  def __init__(self, is_training, config):
    self.maxlen = maxlen = config.maxlen
    dim_proj = config.dim_proj
    n_words = config.n_words
    ydim = config.ydim  # number of classes

    self._input_data = tf.placeholder(tf.int32, [None, maxlen])
    self._mask_data = tf.placeholder(tf.float32, [None, maxlen])
    self._targets = tf.placeholder(tf.int64, [None])

    # make batch size flexible
    self._batch_size = batch_size = tf.placeholder(tf.int32, [])

    # length of sequences for current batch
    self._time_steps = tf.placeholder(tf.int32, [])

    # new lstm cell than has masking
    lstm_cell = rnn_cell.BasicLSTMCellMask(dim_proj)
    if is_training and config.keep_prob < 1:
      lstm_cell = \
          rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

    self._initial_state = lstm_cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [n_words, dim_proj])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only. In general,
    # the rnn() function should be updated with mask option and be used.
    zero_output = array_ops.zeros(array_ops.pack([batch_size, dim_proj]),
                                  tf.float32)
    output = array_ops.zeros(array_ops.pack([batch_size, dim_proj]),
                             tf.float32)
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(config.maxlen):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        call_cell = lambda: lstm_cell(inputs[:, time_step, :], state,
                                      self._mask_data[:, time_step])
        empty_update = lambda: (zero_output, state)
        (cell_output, state) = \
            control_flow_ops.cond(time_step < self._time_steps, call_cell,
                                  empty_update)

        output = tf.add(output, cell_output)  # (adding 0 for short sentences)

    # find sum of the representations
    self._outputs_sum = output

    # find averaged feature of each sentence (shape: batch_size*dim_proj)
    sent_rep = tf.div(output, tf.expand_dims(tf.reduce_sum(self._mask_data,
                                             reduction_indices=1), 1))
    self._sent_rep = sent_rep
    softmax_w = tf.get_variable("softmax_w", [dim_proj, ydim])
    softmax_b = tf.get_variable("softmax_b", [ydim])
    self._logits = tf.matmul(self._sent_rep, softmax_w) + softmax_b
    self._cost = cost = \
        tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self._logits,
                                                           self._targets))
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads = tf.gradients(cost, tvars)
    self._grads = grads

    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def mask_data(self):
    return self._mask_data

  @property
  def targets(self):
    return self._targets

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def time_steps(self):
    return self._time_steps

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def logits(self):
    return self._logits

  @property
  def outputs_sum(self):  # sum of outputs of lstm
    return self._outputs_sum

  @property
  def sent_rep(self):  # sentence representation
    return self._sent_rep

  @property
  def grads(self):
    return self._grads


def get_batches_idx(n, batch_size, shuffle=False):
    """Get the indices of samples for each batch.
    Args:
      n: sample size, int.
      batch_size: batch size, int.
      shuffle: flag to perform shuffling, boolean
    Returns:
      A list where each element contains the batch number and batch indices.
    """
    idx_list = np.arange(n)
    if shuffle:
        np.random.shuffle(idx_list)

    batches = []
    batch_start = 0
    for i in range(n // batch_size):
        batches.append(idx_list[batch_start:batch_start + batch_size])
        batch_start += batch_size

    # Make a batch out of what is left
    if batch_start != n:
       batches.append(idx_list[batch_start:])

    return zip(range(len(batches)), batches)


def prepare_data(seqs, labels, maxlen=None):
  """Create numpy arrays from the seqs.
  This pad each sequence to the same lenght: the length of the
  longest sequence or maxlen. If maxlen is set, it will throw away sequences
  that are longer and pads shorter sequences to have length of maxlen. If
  maxlen is None, it will pad all sequences to have the same length as the
  longest sequence in the batch.
  Args:
    seqs: a list containing the sequences and their correspondin labels
    labels: a list of integer containing the labels for each seq. in seqs.
    maxlen: an integer
  Returns:
    4d list containing:
     x: 2D numpy array containing batch data where each row is a sample of the
        batch
     mask: 2D numpy array containing 1 and 0 with the same size as x
     y: 1D numpy array containing the labels of the samples
     time_steps: length of the max sequence of the batch before masking
    """
  lengths = [len(s) for s in seqs]
  if maxlen is not None and any(item > maxlen for item in lengths):
    new_seqs = []
    new_labels = []
    new_lengths = []
    for l, s, y in zip(lengths, seqs, labels):
      if l <= maxlen:
        new_seqs.append(s)
        new_labels.append(y)
        new_lengths.append(l)
    lengths = new_lengths
    labels = new_labels
    seqs = new_seqs

  if len(lengths) < 1:
    return None, None, None, None

  n_samples = len(seqs)
  maxlen_batch = np.max(lengths)
  seqsize = maxlen if maxlen is not None else maxlen_batch

  x = np.zeros((n_samples, seqsize)).astype('int32')
  x_mask = np.zeros((n_samples, seqsize)).astype('float32')
  for idx, s in enumerate(seqs):
    x[idx, :lengths[idx]] = s
    x_mask[idx, :lengths[idx]] = 1.

  return x, x_mask, labels, maxlen_batch


def prepare_batch_data(data, batch_index, maxlen, training=True):
  """ Get the data in numpy.ndarray format. This makes all data in each batch
  to have the same size (a multiple of maxlen) by adding 0 for short sequences.
  If training is True, all sequences of a batch will have the length of maxlen
  and sequences that have length greater than maxlen will be ignored. Thus,
  a batch size may change.
    If training is False, all sequences of a batch will have the length of
  a*maxlen where a is an integer and its value is determined based on the
  maximum length of the sequences within the batch.
  Args:
    data: a list containing the sequences and their correspondin labels
    batch_index: an integer array containing the indices of the batch.
    maxlen: an integer
    training: boolean
  Returns:
    4d list containing:
     x: 2D numpy array containing batch data where each row is a sample of the
        batch
     mask: 2D numpy array containing 1 and 0 with the same size as x
     y: 1D numpy array containing the labels of the samples
     time_steps: length of the max sequence of the batch before masking
  """
  y = [data[1][t] for t in batch_index]
  x = [data[0][t] for t in batch_index]
  if not training:
    lengths = [len(s) for s in x]
    maxlen = (max(lengths) // maxlen + 1) * maxlen
  x, mask, y, time_steps = prepare_data(x, y, maxlen=maxlen)
  return x, mask, y, time_steps


def evaluate(logits, targets):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    targets: targets tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  correct = tf.nn.in_top_k(logits, targets, 1)
  num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
  return num_correct


def time_epoch(session, m, data, config):
  """Run one epoch on the given data."""
  batch_num = 0.0
  forward_time_epoch = 0.0
  forward_backward_time_epoch = 0.0

  # Get new shuffled index for the training set
  kf = get_batches_idx(len(data[0]), config.batch_size, shuffle=False)

  for step, batch_index in kf:
    # prepare the data for current batch
    x, mask, y, maxlen_batch = prepare_batch_data(data, batch_index,
                                                  config.maxlen, training=True)
    if x is not None:
      start_time = time.time() * 1000
      cost = session.run([m.cost],
                         {m.input_data: x,
                          m.mask_data: mask,
                          m.targets: y,
                          m.time_steps: maxlen_batch,
                          m.batch_size: x.shape[0],
                          })
      forward_time_epoch += time.time() * 1000 - start_time
      start_time = time.time() * 1000
      grads = session.run(m.grads,
                          {m.input_data: x,
                           m.mask_data: mask,
                           m.targets: y,
                           m.time_steps: maxlen_batch,
                           m.batch_size: x.shape[0]
                           })
      forward_backward_time_epoch += time.time() * 1000 - start_time
      batch_num += 1

  return forward_time_epoch / batch_num, \
         forward_backward_time_epoch / batch_num


def main(unused_args):
  config = Config()

  # The data is similar to IMDB review train data with maximum sequence length
  # of 100 which can be found at
  # http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl
  print 'Prepare data'
  seqs_lengths = np.genfromtxt("../../data/seqs_lengths.csv")
  num_samples = len(seqs_lengths)
  seqs = [np.random.randint(1, config.n_words, item) for item in seqs_lengths]
  labels = np.random.randint(config.ydim, size=num_samples)
  data = [seqs, labels]

  if config.CPU_only:
    device_config = tf.ConfigProto(device_count={'GPU': 0},
                               intra_op_parallelism_threads=config.num_threads)
  else:
    device_config = tf.ConfigProto(device_count={'GPU': 1})
  print device_config

  with tf.Graph().as_default(), tf.Session(config=device_config) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m_train = LSTMNet(is_training=True, config=config)

    tf.initialize_all_variables().run()
    forward_time_epochs = np.zeros(config.num_timing_epochs)
    forward_backward_time_epochs = np.zeros(config.num_timing_epochs)
    num_epochs = config.num_warmup_epochs + config.num_timing_epochs
    for i in range(num_epochs):
      if i >= config.num_warmup_epochs:
        # update learning rate for next epoch
        forward_time, forward_backward_time = time_epoch(session, m_train,
                                                         data, config)
        forward_time_epochs[i-config.num_warmup_epochs] = forward_time
        forward_backward_time_epochs[i-config.num_warmup_epochs] \
            = forward_backward_time
      else:  # dry run
        _ = time_epoch(session, m_train, data, config)

    print("time forward: %.4f +- %.4f ms, batch size: %d" %
          (np.mean(forward_time_epochs), np.std(forward_time_epochs),
           config.batch_size))
    print("time gradient: %.4f +- %.4f ms, batch size: %d" %
          (np.mean(forward_backward_time_epochs),
           np.std(forward_backward_time_epochs), config.batch_size))

if __name__ == "__main__":
  tf.app.run()
