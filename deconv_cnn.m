function [ output ] = deconv_cnn( img, ker, net, nsr)
[h, w, ~] = size(img);

if nsr < 0
    img2(:,:,1) = medfilt2(img(:,:,1),[3 3]);
    img2(:,:,2) = medfilt2(img(:,:,2),[3 3]);
    img2(:,:,3) = medfilt2(img(:,:,3),[3 3]);
    enoise = var(img2(:)-img(:));
    evar=(9*var(img(:)))^(0.5)/8;%var(img(:));
    ensr = enoise/evar;
else
    ensr = nsr;
end
tic
img_wnr = deconvwnr(img, ker, ensr);
toc

net.blobs('data').reshape([h w 3 1]);
tic
result_img = net.forward({img_wnr});
toc
output = result_img{1};

end

