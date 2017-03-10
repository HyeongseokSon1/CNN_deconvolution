function [output] = postprocessing( img, ker, result_img, alpha)

output = gpuArray(zeros(size(img)));
for i = 1:3
    y = gpuArray(img(:,:,i));
    y2 = gpuArray(result_img(:,:,i));
    f = gpuArray(ker);   
%     y = img(:,:,i);
%     y2 = result_img(:,:,i);
%     f = ker;   
    sx = size(y);
    sfft = sx;
    
    % deconvolution
    sf = size(f);
    pad_size = sfft - sf;
    f = padarray(f, pad_size, 'post');
    f = fftn(circshift(f, -floor(sf/2)));
    y = fftn(y, sfft);
    y2 = fftn(y2, sfft);
    output(:,:,i) = ifftn(...
        (conj(f).*y + alpha*y2) ./ (conj(f).*f + alpha), sx, 'symmetric');
end
output = gather(output);
end