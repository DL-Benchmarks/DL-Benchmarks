For other layers run e.g. layer 2: replace 'AE1' with 'AE1'. To run the finentuning: replace 'AE1_train_test' by 'SAE_train_test'. For the smaller AE (layer 400, 200, 100) add 'small' in the beginning.

to run timing on cpu (n threads only controllable once during setup) for 100 iterations do:
~/git/caffe/build/tools/caffe time -model ./AE1_train_test.prototxt -iterations 100

to run timing on gpu
~/git/caffe/build/tools/caffe time -model ./AE1_train_test.prototxt -iterations 1000 -gpu 0
