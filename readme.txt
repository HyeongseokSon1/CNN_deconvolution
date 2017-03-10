Author: Hyeongseok Son

This code is an implementation of the paper (Fast Non-blind Deconvolution via Regularized Residual Networks with Long/Short Skip-Connections, ICCP 2017).

Requirement: Matlab, CUDA<=7.5, Caffe
Tested OS: Ubuntu 14.04
Installation procedure:

1. build caffe
- $ cd caffe
- modify parameters and paths in Makefile.config
- $ make all caffe
- $ make matcaffe

2. set the path of matcaffe in demo.m

3. run demo.m


This code includes only a test function.

