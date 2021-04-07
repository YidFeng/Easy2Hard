clc;
clear all;

addpath 'algorithms'
addpath 'graphAnalysisToolbox-1.0'

fileDir = 'C:/Users/dan/Desktop/TNNLS-resubmit-all/realGTfrom/';
outputDir = 'C:/Users/dan/Desktop/TNNLS-resubmit-all/compare_codes/results/SDF/';
inDir = dir(fileDir);
lin = length(inDir);

outDir = dir(outputDir);
lout = length(outDir);
inidx = [1:1:lin];

for m=1:lin
    inname = inDir(m).name;
    for n=1:lout
        outname = outDir(n).name;
        if(strcmp(inname, outname))
            inidx(m)=0;
        end
    end
end
inidx(inidx==0)=[];
for i=1:length(inidx)
    idx = inidx(i);
    path_in = strcat(fileDir,inDir(idx).name);
    f = im2double(imread(path_in)); %input image
    [m,n,c] = size(f);

    %scale-space filtering using joint static and dynamic guidance 
    nei= 1;                   % 0: 4-neighbor 1: 8-neighbor
    lambda = 30;   % regularization parameter
    mu = 50;                  % bw for static guidance
    nu = 400;                 % bw for dymadic guidance
    step = 10;                % num of steps
    issparse = false;

    g=f;            %static guidance image 
    u0=ones(m,n,c); %dynamic guidance image (constant initialization, u0 = 1)

%     disp('===========================================');
%     disp('joint static and dynamic guidance filtering');
%     disp('===========================================');
%     disp('*configuration for scale-space filtering*');
%     St = sprintf(' img: %s \n u^0: costant \n mu: %d \n nu: %d \n isSparse: %d', img_name, mu, nu, issparse);
%     disp(St);fprintf('\n');
    %% joint static and dynamic guidance filtering 
    tic;
    u = sdfilter(g,u0,f,nei,lambda,mu,nu,step,issparse);
    toc;
    fprintf('\n');
    path_out = strcat(outputDir,inDir(idx).name);
    imwrite(u,path_out);
end
    % 
% %scale-space filtering using static guidance (WLS)
% nei= 1;                   % 0: 4-neighbor 1: 8-neighb
% lambda = [1e3 3e5 7e6];   % regularization parameter
% mu = 400;                 % bw for static guidance
% 
% g=f;                      %static guidance image 
% 
% disp('===============================');
% disp('static guidance filtering (WLS)');
% disp('===============================');
% St = sprintf('configuration, img: %s, mu: %d', img_name, mu);
% disp(St);fprintf('\n');
% 
% for l=1:numel(lambda) 
%     result_name = ['wls-filtering_' int2str(lambda(l)) '.png'];
%     
%     %% wls filtering 
%     tic;
%     u = wlsfilter(g,f,nei,lambda(l),mu);
%     toc;
%     fprintf('\n');
%     
%     imwrite(u,fullfile(folder_name,result_name));   
% end



