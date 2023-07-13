clear;close all;
n_superpixels=[48,24,12,6];

save_path='F:\dataset\SAR\sar_reshape_segments.mat';
path = 'F:\dataset\SAR\sar_reshape.png';

I = imread(path);
I=double(I);
[h, w] = size(I);
I = reshape(I, 1, h*w);

I=(mapminmax(I)+1)/2*255;
I=reshape(uint8(I),[h,w,1]);

I = repmat(I, [1,1,3]); % 单通道变三通道

%     gaussian滤波: 用于图像模糊化（去除细节和噪声）
I = imfilter(I, fspecial('gaussian',[5,5]), 'replicate');

E=uint8(zeros([h,w]));

% fine detail structure
sh=SuperpixelHierarchyMex(I,E,0.0,0.1);
segmentmaps=zeros(size(n_superpixels,2),h,w);
for j=1:size(n_superpixels,2)
    GetSuperpixels(sh,n_superpixels(:,j));
    segmentmaps(j,:,:)=sh.label;
end
save( save_path, 'segmentmaps' );




