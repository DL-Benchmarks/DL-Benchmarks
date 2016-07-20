#!/bin/bash
 
mkdir -p outputs

caffe time -model alexnet/train_val.prototxt -iterations 30 -gpu 0  --logtostderr=1 >> outputs/alexnet.log 2>&1
caffe time -model alexnet/train_val_noLRN_noGrouping.prototxt -iterations 30 -gpu 0  --logtostderr=1 >> outputs/alexnet_noLRN_noGrouping.log 2>&1
caffe time -model LeNet/lenet_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/lenet.log 2>&1
caffe time -model stackedAE/AE1_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/large_AE1.log 2>&1
caffe time -model stackedAE/AE2_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/large_AE2.log 2>&1
caffe time -model stackedAE/AE3_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/large_AE3.log 2>&1
caffe time -model stackedAE/SAE_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/large_SAE.log 2>&1
caffe time -model stackedAE/smallAE1_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/small_AE1.log 2>&1
caffe time -model stackedAE/smallAE2_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/small_AE2.log 2>&1
caffe time -model stackedAE/smallAE3_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/small_AE3.log 2>&1
caffe time -model stackedAE/smallSAE_train_test.prototxt -iterations 100 --logtostderr=1 >> outputs/small_SAE.log 2>&1
