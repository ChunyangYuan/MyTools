clear; close all;

FLAG=0;
switch(FLAG)
    case 0
        % hsi=load('..\dataset\Raw_AIR-PolarSAR-Seg\AIR-PolarSAR-Seg-1.mat');hsi=hsi.data;
        hsi=imread('..\dataset\sar_11_rgb.png');
        n_superpixels=[2048,1024,512,256];
    case 1
        hsi=load('..\HyperImage_data\indian\Indian_pines_corrected.mat');hsi=hsi.indian_pines_corrected;
        n_superpixels=[2048,1024,512,256];
        % n_superpixels=[4096,2048,1024,512,256];
    case 2
        hsi=load('..\HyperImage_data\paviaU\PaviaU.mat');hsi=hsi.paviaU;
        n_superpixels=[2048,1024,512,256];
    case 3
        hsi=load('..\HyperImage_data\Salinas\Salinas_corrected.mat');hsi=hsi.salinas_corrected;
        n_superpixels=[2048,1024,512,256];
    case 4
        hsi=load('..\HyperImage_data\KSC\KSC.mat');hsi=hsi.KSC;
        n_superpixels=[2048,1024,512,256];
    case 5
        hsi=load('..\HyperImage_data\Houston2013\Houston.mat');hsi=hsi.Houston;
        n_superpixels=[2048,1024,512,256];
    case 6
        hsi=load('..\HyperImage_data\HyRANK\Loukia.mat');hsi=hsi.Loukia;
        n_superpixels=[2048,1024,512,256];
    case 7
        hsi=load('..\HyperImage_data\Botswana\Botswana.mat');hsi=hsi.Botswana;
        n_superpixels=[2048,1024,512,256];
    case 8
        hsi=load('..\HyperImage_data\Houston2018\HoustonU.mat');hsi=hsi.houstonU;
        n_superpixels=[2048,1024,512,256];
        hsi=hsi(:,1:400,:);%
    case 9
        hsi=load('..\HyperImage_data\xuzhou\xuzhou.mat');hsi=hsi.xuzhou;
        n_superpixels=[2048,1024,512,256];
    case 10
        hsi=load('..\HyperImage_data\WDC\WDC.mat');hsi=hsi.wdc;
        n_superpixels=[2048,1024,512,256];
end


hsi=double(hsi);
[h,w,c]=size(hsi);
% mapminmax: 按行映射到[-1,1]区间内
hsi=mapminmax( reshape(hsi,[h*w,c])');
hsi=reshape(hsi',[h,w,c]);
% gaussian滤波: 用于图像模糊化（去除细节和噪声）
hsi = imfilter(hsi, fspecial('gaussian',[5,5]), 'replicate');
hsi=reshape(hsi,[h*w,c])';% hsi.shape=[c,h*w]
pcacomps=pca(hsi); % pcacomps.shape=[21025, 199]
I=pcacomps(:,[3,2,1])';% I.shape=[3, 21025]
I=(mapminmax(I)+1)/2*255;
I=reshape(uint8(I)',[h,w,3]);
for i=1:3
    I(:,:,i)=imadjust(histeq(I(:,:,i))); %调整图像对比度
end
I = imfilter(I, fspecial('unsharp',0.05), 'replicate'); % I.shape=[145,145,3]


% [h,w,c]=size(hsi);
% I = uint8(hsi);

E=uint8(zeros([h,w]));

% fine detail structure
tic; sh=SuperpixelHierarchyMex(I,E,0.0,0.1); toc
segmentmaps=zeros(size(n_superpixels,2),h,w);
for i=1:size(n_superpixels,2)
    GetSuperpixels(sh,n_superpixels(:,i));
    segmentmaps(i,:,:)=sh.label;
end

switch(FLAG)
    case 0
        save sar_11.mat segmentmaps
    case 1
        save _indian.mat segmentmaps
    case 2
        save segmentmapspaviau.mat segmentmaps
    case 3
        save segmentmapssalinas.mat segmentmaps
    case 4
        save segmentmapsksc.mat segmentmaps
    case 5
        save segmentmapshst.mat segmentmaps
    case 6
        save segmentmapsloukia.mat segmentmaps
    case 7
        save segmentmapsbot.mat segmentmaps
    case 8
        save segmentmapshstu.mat segmentmaps
    case 9
        save segmentmapsxuzhou.mat segmentmaps
    case 10
        save segmentmapswdc.mat segmentmaps
end

% get whatever you want
GetSuperpixels(sh,n_superpixels(1)); color1 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,n_superpixels(2)); color2 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,n_superpixels(3)); color3= MeanColor(double(I),sh.label);
GetSuperpixels(sh,n_superpixels(4)); color4= MeanColor(double(I),sh.label);
% GetSuperpixels(sh,n_superpixels(5)); color5= MeanColor(double(I),sh.label);
% GetSuperpixels(sh,n_superpixels(5)); color6= MeanColor(double(I),sh.label);
% figure,imshow([color1,color2; color3,color4;color5,color6]);
figure,imshow([color1,color2; color3,color4]);