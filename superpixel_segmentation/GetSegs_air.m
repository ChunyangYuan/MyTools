clear; close all;

% n_superpixels=[2048,1024,512,256];
n_superpixels=[4096,1024,256,64]; %16_4
% n_superpixels=[4096,2048,1024,512]; %16_2
% n_superpixels=[16384,4096,1024,256]; %4_4
all_time = 0.0;
% AIR-PolarSAR-Seg-1_1.mat
path = "F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\images_HH_ecb\AIR-PolarSAR-Seg-";
% path = "F:\dataset\Raw_AIR-PolarSAR-Seg\abc\AIR-PolarSAR-Seg-";
save_dir = "F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\segments_HH_16_4_ecb\AIR-PolarSAR-Seg-";
for i=1:500
    for j=1:4
        num_1 = num2str(i);
        num_2 = num2str(j);
        mat_path = strcat(path, num_1, '_HH_', num_2, '.tiff');
        I = imread(mat_path);
        I=double(I);
        [h, w] = size(I);
        I = reshape(I, 1, h*w);
        % [h,w,c]=size(I);
        % % mapminmax: 按行映射到[-1,1]区间内
        % I=mapminmax( reshape(I,[h*w,c])');
        % I=reshape(I',[h,w,c]);
        % gaussian滤波: 用于图像模糊化（去除细节和噪声）
        % I = imfilter(I, fspecial('gaussian',[5,5]), 'replicate');
        % I=reshape(I,[h*w,c])';% I.shape=[c,h*w]
        % pcacomps=pca(I); % pcacomps.shape=[21025, 199]
        % I=pcacomps(:,[3,2,1])';% I.shape=[3, 21025]
        
        I=(mapminmax(I)+1)/2*255;
        I=reshape(uint8(I),[h,w,1]);
        
        I = repmat(I, [1,1,3]); % 单通道变三通道
        
        
        
        % gaussian滤波: 用于图像模糊化（去除细节和噪声）
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
        for ii=1:size(n_superpixels,2)
            GetSuperpixels(sh,n_superpixels(:,ii));
            segmentmaps(ii,:,:)=sh.label;
        end

        % save
        save_path = strcat(save_dir, num_1, "_HH_", num_2, ".mat");
        save( save_path, 'segmentmaps' );
    end

end
% save all_time
filename = 'segment_time_HH.txt';
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




% % get whatever you want
% GetSuperpixels(sh,n_superpixels(1)); color1 = MeanColor(double(I),sh.label);
% GetSuperpixels(sh,n_superpixels(2)); color2 = MeanColor(double(I),sh.label);
% GetSuperpixels(sh,n_superpixels(3)); color3= MeanColor(double(I),sh.label);
% GetSuperpixels(sh,n_superpixels(4)); color4= MeanColor(double(I),sh.label);
% % GetSuperpixels(sh,n_superpixels(5)); color5= MeanColor(double(I),sh.label);
% % GetSuperpixels(sh,n_superpixels(5)); color6= MeanColor(double(I),sh.label);
% % figure,imshow([color1,color2; color3,color4;color5,color6]);
% figure,imshow([color1,color2; color3,color4]);