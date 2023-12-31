clear;close all;
% n_superpixels=[16384,4096,1024,256]; %4_4
% n_superpixels=[8192,2048,512,128]; %8_4
% n_superpixels=[4096,1024,256,64]; %16_4
% n_superpixels=[2048,512,128,32]; %32_4
% n_superpixels=[1024,256,64,16]; %64_4
% n_superpixels=[512,128,32,8]; %128_4
n_superpixels=[256,64,16,4]; %256_4
all_time = 0.0;
dir='F:\Dataset\SAR\sar_output_folder\sar\sar_s';
extension = '.png';
save_dir='F:\Dataset\SAR\sar_output_folder\segments_256\';
%批处理图片进行超像素分割
for i=1:225
    
    num = num2str(i,'%03d');
    path = [dir,num,extension];

%     hsi = imread(path);
%     [h,w,c]=size(hsi);
%     I = uint8(hsi);
%     E=uint8(zeros([h,w]));
    I = imread(path);
    I=double(I);
    [h, w] = size(I);
    I = reshape(I, 1, h*w);

    I=(mapminmax(I)+1)/2*255;
    I=reshape(uint8(I),[h,w,1]);

    I = repmat(I, [1,1,3]); % 单通道变三通道

%     gaussian滤波: 用于图像模糊化（去除细节和噪声）
    I = imfilter(I, fspecial('gaussian',[5,5]), 'replicate');
%     I = imfilter(I, fspecial('laplacian'), 'replicate');
%     I = imfilter(I, fspecial('log'), 'replicate');


    % for ii=1:3
    %     I(:,:,ii)=imadjust(histeq(I(:,:,ii))); %调整图像对比度
    % end
    % I = imfilter(I, fspecial('unsharp',0.05), 'replicate'); % I.shape=[145,145,3]

    E=uint8(zeros([h,w]));

    % fine detail structure
    tic = clock; sh=SuperpixelHierarchyMex(I,E,0.0,0.1); toc = clock;
    all_time = all_time + etime(toc, tic);
    segmentmaps=zeros(size(n_superpixels,2),h,w);
    for j=1:size(n_superpixels,2)
        GetSuperpixels(sh,n_superpixels(:,j));
        segmentmaps(j,:,:)=sh.label;
    end
    save_path = [save_dir,'segments_',num,'.mat'];
    save( save_path, 'segmentmaps' );
end

% save all_time
filename = 'sar_segment_time.txt';
fid = fopen(filename, 'a');
fprintf(fid,'%.10f\n', all_time);
fprintf(fid,'================\n');
% for i=1:10
%     tic = clock;
%     pause(i*0.05)
%     toc = clock;

%     all_time = all_time + etime(toc,tic);
%     fprintf(fid,'%.10f\n', all_time);
%     % fprintf(fid,'\r\n');  % 换行
% end
% fprintf(fid,'================\n');
fclose(fid);



