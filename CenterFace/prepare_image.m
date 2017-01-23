function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels

% mean_data = d.mean_data;
% IMAGE_DIM = 224; 

imgSize = [112, 96];
% coord5points = [0, 0 , 96,96,; ...
%                 8, 104 ,0,104];
% facial5points = [0, 0, size(im,1),size(im,1); ...
%                 0, size(im,2), 0,size(im,2)];
% Tfm =  cp2tform(facial5points', coord5points', 'similarity'); %#ok<DCPTF>
% cropImg = imtransform(im, Tfm, 'XData', [1 imgSize(2)],...
%                                   'YData', [1 imgSize(1)], 'Size', imgSize);
                              
cropImg = imresize(im, [112 96]);  % resize im_data


cropImg = single(cropImg);
cropImg = (cropImg - 127.5)/128;
cropImg = permute(cropImg, [2,1,3]);
crops_data = cropImg(:,:,[3,2,1]);

% crops_data = (im_data - mean_data)/128;  % subtract mean_data (already in W x H x C, BGR)

% % oversample (4 corners, center, and their x-axis flips)
% crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 2, 'single');
% % crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 1, 'single');
% indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
% % n = 1;
% % for i = indices
% %   for j = indices
% %     crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
% %     crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
% %     n = n + 1;
% %   end
% % end
% center = floor(indices(2) / 2) + 1;
% % crops_data(:,:,:,5) = ...
% crops_data(:,:,:,1) = ...
%   im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
% crops_data(:,:,:,2) = crops_data(end:-1:1, :, :, 1);
