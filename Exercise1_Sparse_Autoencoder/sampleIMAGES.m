function patches = sampleIMAGES(imSize ,nPatches)
% Returns  images patches for training.
%
% imSize: # of pixels per edge of square patch
% nPatches: # of training patches to sample

% Data Source:
% IMAGES is a 3D array containing 10 images
% For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
% and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
% it. (The contrast on these images look a bit off because they have
% been preprocessed using using whitening. As a second example, 
% IMAGES(21:30,21:30,1) is an image patch corresponding to the pixels in 
% the block (21,21) to (30,30) of Image 1

% load images from disk 
load IMAGES;    

% initialize parameters
patches = zeros(imSize*imSize, nPatches);
randImage = randi(10,nPatches,1); % get image #
randRowPixel = randi(505,nPatches,1); % get starting row number
randColPixel = randi(505,nPatches,1); % get starting column number

% sample IMAGES
for i=1:10000
    A = IMAGES(randRowPixel(i):randRowPixel(i)+(imSize-1), ...
        randColPixel(i):randColPixel(i)+(imSize-1),randImage(i));
    patches(:,i) = A(:); % convert image patches to vectors
end

% normalize data to [0,1]
patches = normalizeData(patches);
end

function patches = normalizeData(patches)
% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
end