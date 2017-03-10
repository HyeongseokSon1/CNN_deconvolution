clear all, close all;
addpath('caffe/matlab'); %require matcaffe path
weights = 'model/paper.caffemodel';
model = 'model/net.prototxt';

%% load an image and a blur kernel

%% Fig.8 in main paper
img = imread('images/kodim07.png');
ker = im2double(imread('images/kernel_e.png'));
nsr = -1;
alpha = 0.5;
noise_var =(0.01)^2; 

%% Fig.9 in main paper
% img = imread('images/kodim11.png');
% ker = im2double(imread('images/kernel_d.png'));
% nsr = -1;
% alpha = 2.0;
% noise_var =(0.03)^2;

img = im2double(img);
ker = ker(:,:,1) / (sum(sum(ker(:,:,1))));

gt = img;
[h, w, ~] = size(gt);
[kh,kw] = size(ker);
khh = round(0.5*kh);
kwh = round(0.5*kw);
xest = gpuArray(zeros(h, w, 3));

%% make an synthetic blurred image
img = imfilter(img, ker, 'circular','conv');
img = imnoise(img, 'gaussian', 0, noise_var); 

%% run deconv_cnn
caffe.set_mode_gpu();
net = caffe.Net(model, weights, 'test');
result_img = deconv_cnn(img,ker,net,nsr); % if nsr < 0, it uses estimated nsr
caffe.reset_all();

psnr1= psnr(double(result_img(1+khh:end-khh,1+kwh:end-kwh,:)), gt(1+khh:end-khh,1+kwh:end-kwh,:)); %exclude a boundary 

%% post processing
tic
xest = postprocessing(img, ker, result_img,alpha);
toc

psnr2 = psnr(double(xest(1+khh:end-khh,1+kwh:end-kwh,:)), gt(1+khh:end-khh,1+kwh:end-kwh,:)); %exclude a boundary 
imwrite(xest,'out.png');