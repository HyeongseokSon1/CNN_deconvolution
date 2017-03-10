# CNN_deconvolution
Unoptimized implementation of <b>"Fast Non-blind Deconvolution via Regularized Residual Networks with Long/Short Skip-Connections"</b> in Matlab and Caffe.

## How to use it
Requirement: Matlab, CUDA<=7.5, Caffe<br>
Tested OS: Ubuntu 14.04<br>
Installation procedure:<br>

1. build caffe <br>
$ cd caffe <br>
-modify parameters and paths in Makefile.config <br>
$ make all caffe <br>
$ make matcaffe <br>
2. set the path of matcaffe in demo.m <br>
3. run demo.m <br>
* This code includes only a test function.

## Contributors
Hyeongseok Son (sonhs@postech.ac.kr)

## Citation
Cite our papers if you find this software useful.<br>
1. Hyeongseok Son, Seungyong Lee, "[Fast Non-blind Deconvolution via Regularized Residual Networks with Long/Short Skip-Connections]
(http://cg.postech.ac.kr/research/cnn_deconvolution/)", IEEE International Conference on Computational Photography (ICCP) 2017, 2017. 







