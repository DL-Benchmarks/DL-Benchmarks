# DL-benchmarks

This is the companion code for DL benchmarking study reported in the paper *Comparative Study of Deep Learning Software Frameworks* by *Soheil Bahrampour, Naveen Ramakrishnan, Lukas Schott, and Mohak Shah*. The paper can be found here http://arxiv.org/abs/1511.06435. The code allows the users to reproduce and extend the results reported in the study. The code provides timings of forward run and forward+backward (gradient computation) run of several deep learning architecture using Caffe, Neon, TensorFlow, Theano, and Torch. The deep learning architectures used includes LeNet, AlexNet, LSTM, and a stacked AutoEncoder. Please cite the above paper when reporting, reproducing or extending the results.


## Run the benchmarks
See the readme file within each folder to run the experiments. 


## Dependencies
Please refer to the corresponding github repository of each framework to install the framework and the corresponding dependencies. Most of the packages require Nvidia cuda and cuDNN to run on GPU. The provided codes have been tested on a system with Ubuntu 14.04 + Nvidia TitanX GPU with CUDA 7.5 and cuDNNv3.


## License

DL-benchmarks is open-sourced under the MIT license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in DL-benchmarks, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).
