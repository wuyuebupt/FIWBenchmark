clc;
clear;
close all;
% load('pairs/m-s_pairs.mat');
pairs_file = 'relations/gm-gd.mat';
%pairs_file = 'vgg_face/gm-gd_pairs.mat';
scores_file = 'scores/gm-gd.mat';

% save relations/ss_fold5.mat pairs

load(pairs_file);
%% disp pair write to file
% for fold_id = 1:10
% for fold_id = 1:5
for fold_id = 1

fold = cell2mat(pairs(:,1));
index = (fold == fold_id);
%% pca training
index_training = (fold ~= fold_id);
%%
fff = pairs(index,:);
prefix = '/local3/kinship/pami/FIDS/';

% if exist('../+caffe', 'dir')
  addpath('/home/smilelab/projects/faces/ijcai2017/centerface/caffe-face/matlab');
% else
%   error('Please run this demo from caffe/matlab/demo');
% end

% load model
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
% model_dir = '';
net_model = 'VGG_FACE_deploy.prototxt';
% net_weights = strcat('/home/smilelab/project/kinship/caffe/examples/kinship_fs/models_each_fold/',int2str(fold_id),'/vgg_kin__iter_2740.caffemodel');
% net_weights = strcat('/home/smilelab/project/kinship/caffe/examples/kinship_fmcls/vgg_face/VGG_FACE.caffemodel');
% net_weights = strcat('/home/smilelab/project/vgg_face/vgg_face_caffe/VGG_FACE.caffemodel');
net_weights = strcat('./newmodel/',int2str(1),'/vgg_fm_iter_1400.caffemodel');
% net_weights = strcat('./models/',int2str(1),'/finetune_iter_1400.caffemodel');
% net_weights = strcat('./models/','snapshot/finetune_iter_700.caffemodel');

% net_weights = strcat('./newmodel/',int2str(1),'/vgg_fm_iter_1400.caffemodel');
%net_weights = '/home/smilelab/project/kinship/caffe/examples/kinship_fs/models_each_fold/2/vgg_kin_iter_1500.caffemodel';
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('No model file');
end
tic;
net = caffe.Net(net_model, net_weights, phase);
toc;
%d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_value = [129.1863, 104.7624, 93.5840];


%% extract training features
train_fff = pairs(index_training,:);

% allpaths ={};
for i = 1:size(train_fff,1)
    disp(i);
%     allpaths{2*i-1} = [prefix strrep(char(train_fff(i,3)),'-feats.mat','.jpg')];
%     allpaths{2*i} = [prefix strrep(char(train_fff(i,4)),'-feats.mat','.jpg')];
    allpaths{2*i-1} = [prefix char(train_fff(i,3))];
    allpaths{2*i} = [prefix char(train_fff(i,4))];
end    


allpaths_uniq = unique(allpaths, 'rows');
for i = 1:size(allpaths_uniq , 2) 
    disp(i);
    % extract feature
    img = imread(char(allpaths_uniq(i)));
    input_data = {prepare_image1(img, mean_value)};
    % fc7_ft1 = mean(cell2mat(net.forward(input_data)),2); 
    tic;
    fc7_ft1 = cell2mat(net.forward(input_data));
    toc;
    fea_pca{i}  = fc7_ft1/norm(fc7_ft1);
    % = cos_sim(fc7_ft1(:,1), fc7_ft2(:,1));
end

%% extract all features for test
for i = 1:size(fff,1)
    disp(i);
%     allpaths_test{2*i-1} = [prefix strrep(char(fff(i,3)),'-feats.mat','.jpg')];
%     allpaths_test{2*i} = [prefix strrep(char(fff(i,4)),'-feats.mat','.jpg')];
    allpaths_test{2*i-1} = [prefix char(fff(i,3))];
    allpaths_test{2*i} = [prefix char(fff(i,4))];
end
 
allpaths_uniq_test = unique(allpaths_test, 'rows');
for i = 1:size(allpaths_uniq_test , 2)
    disp(i);
    % extract feature
    img = imread(char(allpaths_uniq_test(i)));
    input_data = {prepare_image1(img, mean_value)};
    % fc7_ft1 = mean(cell2mat(net.forward(input_data)),2); 
    tic;
    fc7_ft1 = cell2mat(net.forward(input_data));
    toc;
    feas_test{i}  = fc7_ft1/norm(fc7_ft1);
    % = cos_sim(fc7_ft1(:,1), fc7_ft2(:,1));
end


caffe.reset_all();
%% concentract all features
all_fea = [fea_pca feas_test];

%% pca
tic;
% [eigvec, eigval, ~, sampleMean] = PCA(cell2mat(fea_pca)');
[eigvec, eigval, ~, sampleMean] = PCA(cell2mat(all_fea)');
toc;
% Wdims = size(eigval,1);
Wdims = 100;

%% testing
for i = 1:size(fff, 1)
    disp(i);
    tic;
    label(i) = fff(i,2);
%     path1 = [prefix strrep(char(fff(i,3)),'-feats.mat','.jpg')];
%     path2 = [prefix strrep(char(fff(i,4)),'-feats.mat','.jpg')];
    path1 = [prefix char(fff(i,3))];
    path2 = [prefix char(fff(i,4))];
    toc;
    % extract feature
    % img = imread(path1);
    % input_data = {prepare_image1(img, mean_value)};
    % fc7_ft1 = mean(cell2mat(net.forward(input_data)),2); 
    
    % fc7_ft1 = cell2mat(net.forward(input_data))';
    
    % find the index of path1
    tic;
    index_1 = strfind(allpaths_uniq_test, path1);
    index = find(~cellfun(@isempty, index_1));
    fc7_ft1 = cell2mat(feas_test(index))';
    
    % img = imread(path2);
    % input_data = {prepare_image1(img, mean_value)};
    % fc7_ft2 = cell2mat(net.forward(input_data))';
      
    index_2 = strfind(allpaths_uniq_test, path2);
    index = find(~cellfun(@isempty, index_2));
    fc7_ft2 = cell2mat(feas_test(index))';
    toc;
    
    
    score1(i) = cos_sim(fc7_ft1', fc7_ft2');
    % score1(i) = norm(fc7_ft1' - fc7_ft2');
    tic;
    fc7_ft1 = (bsxfun(@minus, fc7_ft1, sampleMean) * eigvec(:, 1:Wdims));    
    fc7_ft2 = (bsxfun(@minus, fc7_ft2, sampleMean) * eigvec(:, 1:Wdims));
    toc;
    % fc7_ft1 = mean(cell2mat(net.forward(input_data)),2);
    % calculate cosine
    score_pca(i) = cos_sim(fc7_ft1', fc7_ft2');
    % score_pca(i) = norm(fc7_ft1' - fc7_ft2');
    toc;
    % score2(i) = cos_sim(fc7_ft1(:,2), fc7_ft2(:,2));
    % score3(i) = score1(i) + score2(i);
    % 
end

%% 
score1_all{fold_id} = score1;
[fpr, tpr, ~, ~, acc] = ROCcurve(score1, cell2mat(label));
disp(acc);
acc_ori{fold_id} = acc;
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;
% hold on;


score_pca_all{fold_id} = score_pca;
[fpr, tpr, ~, ~, acc] = ROCcurve(score_pca, cell2mat(label));
disp(acc);
acc_pca{fold_id} = acc;
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;

end


save(scores_file, 'score1_all', 'score_pca_all');

%% 
% allscore5_pca = cell2mat(score_pca_all);
% allscore5 = cell2mat(score1_all);
% label5 = pairs(:,2);
% 
% [fpr, tpr, ~, ~, acc] = ROCcurve(allscore5, cell2mat(label5)');
% disp(acc);
% acc_pca_5= acc;
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;
% 
% hold on;
% 
% [fpr, tpr, ~, ~, acc] = ROCcurve(allscore5_pca, cell2mat(label5)');
% disp(acc);
% acc_pca_5= acc;
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;

% 
% [fpr, tpr, ~, ~, acc] = ROCcurve(score3, cell2mat(label));
% disp(acc);
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;

