clear all, close all;
addpath('caffe/matlab'); %require matcaffe path
addpath('liuetal');
weights = 'model/paper.caffemodel';
model = 'model/net.prototxt';

%% load an image and a blur kernel

%% Fig.1 in supplemental material
img = imread('images/kodim08.png');
ker = im2double(imread('images/kernel_a.png'));
nsr = -1;

%% Fig.1 in supplemental material
% img = imread('images/kodim06.png');
% ker = im2double(imread('images/kernel_d.png'));
% nsr = 0.003;

img = im2double(img);
ker = ker(:,:,1) / (sum(sum(ker(:,:,1))));

%% make an synthetic blurred image without padding
img_np(:,:,1) = conv2(img(:,:,1),ker,'valid');
img_np(:,:,2) = conv2(img(:,:,2),ker,'valid');
img_np(:,:,3) = conv2(img(:,:,3),ker,'valid');
noise_var =(0.01)^2;
img_np = imnoise(img_np, 'gaussian', 0, noise_var); 

[kh,kw] = size(ker);
khh = floor(0.5*kh);
kwh = floor(0.5*kw);

gt = img(1+khh:end-khh,1+kwh:end-kwh,:);
[h, w, ~] = size(gt);
xest = gpuArray(zeros(h, w, 3));

%% circular padding via Liu 2008
img_w = wrap_boundary_liu(img_np, opt_fft_size([h w]+size(ker)-1));


%% run deconv_cnn
caffe.set_mode_gpu();
net = caffe.Net(model, weights, 'test');

result_img = deconv_cnn(img_w,ker,net,nsr); % if nsr < 0, it uses estimated nsr
caffe.reset_all();
result_temp = result_img;
result_img = result_img(1:h,1:w,:);
psnr1= psnr(double(result_img(1+khh:end-khh,1+kwh:end-kwh,:)), gt(1+khh:end-khh,1+kwh:end-kwh,:)); 

imwrite(double(result_img),'out.png');
