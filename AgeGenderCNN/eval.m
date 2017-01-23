clc;
clear;
close all;

pairs_file = 'data/gm-gd.mat';
results_file = 'results/gm-gd.mat';

load(pairs_file);


prefix = '/local3/kinship/pami/FIDS/';

% add caffe path
addpath('/home/smilelab/projects/faces/ijcai2017/centerface/caffe-face/matlab');

% load model
caffe.set_mode_gpu();
gpu_id = 0;  
caffe.set_device(gpu_id);
net_model = 'VGG_FACE_deploy.prototxt';

net_weights = strcat('gender.caffemodel');
net_weights = strcat('dex_chalearn_iccv2015.caffemodel');
phase = 'test'; % run with phase test (so that dropout isn't applied)

if ~exist(net_weights, 'file')
  error('No model file');
end
tic;
net = caffe.Net(net_model, net_weights, phase);
toc;

mean_value = [129.1863, 104.7624, 93.5840];


% extract training features
train_fff = pairs;
for i = 1:size(train_fff,1)
    disp(i);
    allpaths{2*i-1} = [prefix char(train_fff(i,3))];
    allpaths{2*i} = [prefix char(train_fff(i,4))];
end    
allpaths_uniq = unique(allpaths, 'rows');
for i = 1:size(allpaths_uniq , 2) 
    disp(i);
    % extract feature
    img = imread(char(allpaths_uniq(i)));
    input_data = {prepare_image(img, mean_value)};
    fc7_ft1 = cell2mat(net.forward(input_data));
    fea_pca{i}  = fc7_ft1/norm(fc7_ft1);
end

caffe.reset_all();
% concentract all features
all_fea = fea_pca;

% pca
tic;
[eigvec, eigval, ~, sampleMean] = PCA(cell2mat(all_fea)');
toc;

Wdims = 100;
fold = cell2mat(pairs(:,1));
for fold_id = 1:5
% for fold_id = 1

index_f = (fold == fold_id);
fff = pairs(index_f,:);

% testing
disp(fold_id);
for i = 1:size(fff, 1)
%     disp(i);
    label(i) = fff(i,2);
    path1 = [prefix char(fff(i,3))];
    path2 = [prefix char(fff(i,4))];
    index_1 = strfind(allpaths_uniq, path1);
    index = find(~cellfun(@isempty, index_1));
    fc7_ft1 = cell2mat(all_fea(index))';
    
      
    % find the index of path2
    index_2 = strfind(allpaths_uniq, path2);
    index = find(~cellfun(@isempty, index_2));
    fc7_ft2 = cell2mat(all_fea(index))';
    
    score(i) = cos_sim(fc7_ft1', fc7_ft2');
    
    fc7_ft1 = (bsxfun(@minus, fc7_ft1, sampleMean) * eigvec(:, 1:Wdims));    
    fc7_ft2 = (bsxfun(@minus, fc7_ft2, sampleMean) * eigvec(:, 1:Wdims));
    
    % calculate cosine after pca
    score_pca(i) = cos_sim(fc7_ft1', fc7_ft2');
end

%
score1_all{fold_id} = score;
[fpr, tpr, ~, ~, acc] = ROCcurve(score, cell2mat(label));
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


save(results_file, 'score1_all', 'score_pca_all', 'acc_ori', 'acc_pca');
