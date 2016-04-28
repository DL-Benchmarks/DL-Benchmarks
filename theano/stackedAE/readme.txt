To run the stacked AE you have the following options:

    hidden_layers_sizes = [400, 200, 100]  # bigger version 800, 1000, 2000
    n_warmup_iter = 400      # number of dry runs before timing
    n_timing_iter = 800      # number of iterations to average over
    batch_size = 64
    n_ins = 28**2            # number pixels input

To run the SAE on cpu:
THEANO_FLAGS='device=cpu' python SAE.py

To change number of threads e.g. 12:
OMP_NUM_THREADS=12 THEANO_FLAGS='device=cpu' python SAE.py

To run on GPU:
THEANO_FLAGS='device=gpu' python SAE.py
