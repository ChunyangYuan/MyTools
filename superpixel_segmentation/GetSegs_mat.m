clear; close all;

n_superpixels=[2048,1024,512,256];
all_time = 0.0;
% AIR-PolarSAR-Seg-1_1.mat
path = "F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\mat\AIR-PolarSAR-Seg-";
% path = "F:\dataset\Raw_AIR-PolarSAR-Seg\abc\AIR-PolarSAR-Seg-";
save_dir = "F:\dataset\Raw_AIR-PolarSAR-Seg\splited_dataset\segments\AIR-PolarSAR-Seg-";
for i=1:500
    for j=1:4
        num_1 = num2str(i);
        num_2 = num2str(j);
        mat_path = strcat(path, num_1, '_', num_2, '.mat');
        hsi = load(mat_path);hsi=hsi.data;
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
        % k = 1
        % size(I)
        I=(mapminmax(I)+1)/2*255;
        % k = 2
        % size(I)
        I=reshape(uint8(I)',[h,w,3]);
        % k = 3
        % size(I)
        for ii=1:3
            I(:,:,ii)=imadjust(histeq(I(:,:,ii))); %调整图像对比度
        end
        % k = 4
        % size(I)
        I = imfilter(I, fspecial('unsharp',0.05), 'replicate'); % I.shape=[145,145,3]
        % k = 5
        % size(I)
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
        save_path = strcat(save_dir, num_1, "_", num_2, ".mat");
        save( save_path, 'segmentmaps' );
    end

end
% save all_time
filename = 'segment_time_AIR_SAR.txt';
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