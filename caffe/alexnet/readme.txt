To run caffe alexnet:

to run timing on cpu (n threads only controllable once during setup) for 30 iterations do:
~/src/benchmarking_dl/proj-rtc13-dl/release/caffe/alexnet$ ~/git/caffe/build/tools/caffe time -model ./train_val.prototxt -iterations 30

To run on GPU:
~/src/benchmarking_dl/proj-rtc13-dl/release/caffe/alexnet$ ~/git/caffe/build/tools/caffe time -model ./train_val.prototxt -iterations 30 -gpu 0

To run without LRN and without grouping
~/src/benchmarking_dl/proj-rtc13-dl/release/caffe/alexnet$ ~/git/caffe/build/tools/caffe time -model ./train_val_noLRN_noGrouping.prototxt -iterations 30 -gpu 0
