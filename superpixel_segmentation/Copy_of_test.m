clear; close all;

n_superpixels=[2048,1024,512,256];
all_time = 0.0;

data_path='F:\Edge\MTS12 dataset\s1\train\';
fileFolder=fullfile(data_path);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
file_name_list={dirOutput.name};

save_dir = "F:\Edge\MTS12 dataset\s1\train_segments\";
for i = 1:length(file_name_list)
    
    file_name=file_name_list{i};%     读取文件名
%     j=find('.'==filename);
    file_name_no_postfix=file_name(1:end-4);
    
    sar_path=strcat(data_path,file_name);
    sar = load(sar_path);%     读取mat文件
%     sar = imread(sar_path);%     读取png,jpg

    sar=double(sar);
    [h,w,c]=size(sar);
    
    % mapminmax: 按行映射到[-1,1]区间内
    sar=mapminmax( reshape(sar,[h*w,c])');
    hsi=reshape(hsi',[h,w,c]);
    
    % gaussian滤波: 用于图像模糊化（去除细节和噪声）
    hsi = imfilter(hsi, fspecial('gaussian',[5,5]), 'replicate');
    hsi=reshape(hsi,[h*w,c])';% hsi.shape=[c,h*w]
    
    sar=(mapminmax(sar)+1)/2*255; %0-255
    sar=reshape(uint8(sar)',[h,w,3]);
    E=uint8(zeros([h,w]));
%     for ii=1:3
%         sar(:,:,ii)=imadjust(histeq(sar(:,:,ii))); %调整图像对比度
%     end
%     sar = imfilter(sar, fspecial('unsharp',0.05), 'replicate'); % sar.shape=[145,145,3]
    % fine detail structure
    tic = clock; sh=SuperpixelHierarchyMex(sar,E,0.0,0.1); toc = clock;
    all_time = all_time + etime(toc, tic);
    segmentmaps=zeros(size(n_superpixels,2),h,w);
    for ii=1:size(n_superpixels,2)
        GetSuperpixels(sh,n_superpixels(:,ii));
        segmentmaps(ii,:,:)=sh.label;
    end

    % save
    save_path = strcat(save_dir, file_name_no_postfix, "_segments.mat");
    save( save_path, 'segmentmaps' );
end

% save all_time
filename = 'segment_time.txt';
fid = fopen(filename, 'a');
fprintf(fid,'%.10f\n', all_time);
% fprintf(fid,'================\n');
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