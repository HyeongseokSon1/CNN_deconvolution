# Introduction
Unoptimized implementation of <b>"Fast Non-blind Deconvolution via Regularized Residual Networks with Long/Short Skip-Connections"</b> in Matlab and Caffe.

This project uses a Convolutional Neural Network (CNN) to improve a performance of non-blind deconvolution. It uses Wiener deconvolution as a pre-deconvolution, and it enables the network to process a general non-blind deconvolution problem by training various blur kernels and noise levels.

For more details regarding this technique, please refer to the paper

* Example
![Example](docs/images/deconv_example.png)  
  * It requires a blur kernel for a blurred image

* Framework
![Framework](docs/images/framework.jpg)

## How to use it
Requirement: Matlab, CUDA<=7.5, Caffe<br>
Tested OS: Ubuntu 14.04<br>
Installation procedure:<br>

1. build caffe <br>
$ cd caffe <br>
-modify parameters and paths in Makefile.config <br>
$ make all <br>
$ make matcaffe <br>
2. set the path of matcaffe in demo.m <br>
3. run demo.m <br>
* This code includes only a test function.

## Contributors
Hyeongseok Son (sonhs@postech.ac.kr)

## Citation
Cite our papers if you find this software useful.<br>
1. Hyeongseok Son, Seungyong Lee, "[Fast Non-blind Deconvolution via Regularized Residual Networks with Long/Short Skip-Connections](http://cg.postech.ac.kr/research/resnet_deconvolution/)", IEEE International Conference on Computational Photography (ICCP) 2017, 2017. 

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition, personalization technology through user preference analysis is under study.

Please checkout out other Coupe repositories in our Posgraph github organization.

## Coupe Project
* [Coupe Website](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)




