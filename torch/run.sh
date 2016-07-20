#!/bin/bash
 
mkdir -p outputs

th LeNet/lenet.lua --logtostderr=1 >> outputs/lenet.log 2>&1
cd alexnet
th alexnet.lua  --logtostderr=1 >> ../outputs/alexnet.log 2>&1
cd ..
th stackedAE/SAE.lua  --logtostderr=1 >> outputs/SAE.log 2>&1
cd lstm
th lstm.lua  --logtostderr=1 >> ../outputs/lstm.log 2>&1
cd ..
