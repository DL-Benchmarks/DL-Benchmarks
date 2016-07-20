#!/bin/bash
 
mkdir -p outputs

python LeNet/lenet.py  --logtostderr=1 >> outputs/lenet.log 2>&1
python alexnet/alexnet.py  --logtostderr=1 >> outputs/alexnet.log 2>&1
python stackedAE/stackedAE.py  --logtostderr=1 >> outputs/SAE.log 2>&1
cd lstm
python lstm.py  --logtostderr=1 >> ../outputs/lstm.log 2>&1
cd ..
