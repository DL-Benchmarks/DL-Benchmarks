to run timing on cpu (n threads only controllable once during setup) for 100 iterations do:
~/git/caffe/build/tools/caffe time -model ./lenet_train_test.prototxt -iterations 100

to run timing on gpu
~/git/caffe/build/tools/caffe time -model ./lenet_train_test.prototxt -iterations 100 -gpu 0

