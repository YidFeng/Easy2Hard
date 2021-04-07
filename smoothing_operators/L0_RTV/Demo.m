% Demo script
% Uncomment each case to see the results 
% for i=145
% i = 21;
% num = num2str(i);
% img = strcat(num,'.jpg');
% path_in = strcat('new23/',img);
% path_out = strcat('detexture2/',img);
% I = (imread(path_in));
%
% warning off;
% addpath(genpath('ext'));
% 
% 
% fileDir = 'C:/Users/dan/Desktop/TNNLS-resubmit-all/realGTfrom/';
% outDir = 'C:/Users/dan/Desktop/TNNLS-resubmit-all/compare_codes/results/RTV/';
% dirOutput = dir(fileDir);
% [~, ind] = sort([dirOutput(:).datenum], 'ascend');
% a = dirOutput(ind); 
% for i=1:length(ind)
%     tic
%     path_in = strcat(fileDir,a(i).name)
%     I = imread(path_in);
%     S = tsmooth(I,0.04,5);
%     S = L0Smoothing(I,0.05);
%     path_out = strcat(outDir,a(i).name);
%     imwrite(S,path_out);
%     toc
% end
 I = (imread('0001.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5)
% S = tsmooth(I,0.01,2);
 S = L0Smoothing(I,0.01);
% figure, imshow(I), figure, imshow(S);
imwrite(S,'0001l.jpg')
% imwrite(S,'clean11.jpg')

% end
% % 
% I = (imread('small2.jpg'));
% S1 = L0Smoothing(I,0.03);
% S2 = tsmooth(I,0.03,4);
% figure, imshow(I),figure, imshow(S1), figure, imshow(S2);


% I = (imread('imgs/crossstitch.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);


% I = (imread('imgs/mosaicfloor.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5); 
% figure, imshow(I), figure, imshow(S);






