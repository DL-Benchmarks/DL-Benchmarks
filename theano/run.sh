#!/bin/bash
 
mkdir -p outputs

THEANO_FLAGS='device=gpu0' python LeNet/lenet.py  --logtostderr=1 >> outputs/lenet.log 2>&1
THEANO_FLAGS='device=gpu0' python alexnet/train.py  --logtostderr=1 >> outputs/alexnet.log 2>&1
THEANO_FLAGS='device=gpu0' python stackedAE/SAE.py  --logtostderr=1 >> outputs/SAE.log 2>&1
cd lstm
THEANO_FLAGS='device=gpu0' python lstm.py  --logtostderr=1 >> ../outputs/lstm.log 2>&1
cd ..
