To run the code on gpu:
THEANO_FLAGS='device=gpu' python SCRIPT.py

To run the code on cpu:
THEANO_FLAGS='device=cpu' python SCRIPT.py

To run on certain number of threads on cpu:
OMP_NUM_THREADS=8 THEANO_FLAGS='device=cpu' python SCRIPT.py 

