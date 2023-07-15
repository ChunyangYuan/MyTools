clear;close all;

% n_superpixels=[2048,1024,512,256];

n_superpixels=[4096,1024,256,64]; %16_4
all_time = 0.0;
dir='F:\Dataset\multi_sensor_landcover_classification\output_folder\munich_s1\Munich_s1_s';
extension = '.tif';
save_dir='F:\Dataset\multi_sensor_landcover_classification\output_folder\munich_segments\';
%批处理图片进行超像素分割
for i=1:624
    
    num = num2str(i,'%03d');
    path = [dir,num,extension];

    I = imread(path);
    I=double(I);
    [h, w] = size(I);
    I = reshape(I, 1, h*w);

    I=(mapminmax(I)+1)/2*255;
    I=reshape(uint8(I),[h,w,1]);

    I = repmat(I, [1,1,3]); % 单通道变三通道

%     gaussian滤波: 用于图像模糊化（去除细节和噪声）
    I = imfilter(I, fspecial('gaussian',[5,5]), 'replicate');

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
filename = 'segment_time_munich_mmcv.txt';
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



