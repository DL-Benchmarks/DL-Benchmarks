# DL-benchmarks

This is the companion code for DL benchmarking study reported in the paper *Comparative Study of Deep Learning Software Frameworks* by *Soheil Bahrampour, Naveen Ramakrishnan, Lukas Schott, and Mohak Shah*. The paper can be found here http://arxiv.org/abs/1511.06435. The code allows the users to reproduce and extend the results reported in the study. The code provides timings of forward run and forward+backward (gradient computation) run of several deep learning architecture using Caffe, Neon, TensorFlow, Theano, and Torch. The deep learning architectures used includes LeNet, AlexNet, LSTM, and a stacked AutoEncoder. Please cite the above paper when reporting, reproducing or extending the results.

# Updated results
Here you can find a set of new timings obtained using **cuDNNv4** on a **single M40 GPU** on the same experiments performed in the paper. The result are reported using **Caffe-Nvidia 0.14.5**, **Tensoflow 0.9.0rc0**, **Theano 0.8.2**, and **Torch7**. 

1) **LeNet** using batch size of 64 (Extension of Table 3 in the paper)

|   Setting  | Gradient (ms) | Forward (ms) |
|:----------:|:-------------:|:------------:|
| Caffe |     2.4      |   0.8        |
| Tensorflow |      2.7      |      0.8     |
|   Theano   |      **1.6**      |      0.6     |
|    Torch   |      1.8      |      **0.5**    |

2) **Alexnet** using batch size of 256 (Extension of Table 4 in the paper)

|   Setting  | Gradient (ms) | Forward (ms) |
|:----------:|:-------------:|:------------:|
| Caffe |      279.3     |      **88.3**     |
| Tensorflow |      **276.6**      |      91.1     |
|    Torch   |     408.8      |      98.8     |

3) **LSTM** using batch size of 16 (Extension of Table 6 in the paper)

|   Setting  | Gradient (ms) | Forward (ms) |
|:----------:|:-------------:|:------------:|
| Tensorflow |      85.4      |      37.1     |
|    Theano   |     **17.3**      |      **4.6**     |

4) **Stacked auto-encoder** with encoder dimensions of 400, 200, 100 using batch size of 64 (Extension of Table 5 in the paper)

|   Setting  | Gradient (ms) AE1 | Gradient (ms) AE2 | Gradient (ms) AE3 | Gradient (ms) Total pre-training | Gradient (ms) SE | Forward (ms) SE |
|:----------:|:-----------------:|:-----------------:|:-----------------:|:--------------------------------:|:----------------:|:---------------:|
| Caffe |       0.8        |    0.9       |      0.9         |          2.6        |       1.1        |       0.6       |
| Tensorflow |        0.7        |        0.6        |        0.6        |                1.9               |        1.2       |       0.4       |
|   Theano   |        0.6        |        0.4        |        0.3        |                **1.3**              |        **0.4**       |       **0.3**       |
|    Torch   |        0.5        |        0.5        |        0.5        |                1.5               |        0.6       |       **0.3**       |

5)  **Stacked auto-encoder** with encoder dimensions of 800, 1000, 2000 using batch size of 64 (Extension of Table 7 in the paper)

|   Setting  | Gradient (ms) AE1 | Gradient (ms) AE2 | Gradient (ms) AE3 | Gradient (ms) Total pre-training | Gradient (ms) SE | Forward (ms) SE |
|:----------:|:-----------------:|:-----------------:|:-----------------:|:--------------------------------:|:----------------:|:---------------:|
| Caffe |         0.9     |      1.2     |        1.7      |          3.8       |     1.9        |       0.9       |
| Tensorflow |        0.9        |        1.1        |        1.6        |                3.6               |        2.1       |       0.7       |
|   Theano   |        0.7        |        1.0        |        1.8        |                3.5               |        **1.2**       |       **0.6**       |
|    Torch   |        0.7        |        0.9        |        1.4        |                **3.0**               |        1.4       |      **0.6**       |


## Run the benchmarks
See the readme file within each folder to run the experiments. 

## License

DL-benchmarks is open-sourced under the MIT license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in DL-benchmarks, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).
