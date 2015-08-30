%% CS294A/CS294W Linear Decoder Exercise
% In this exercise, we implement a linear decoder (a sparse autoencoder
% whose output layer uses a linear activation function). You will then 
% apply it to learn features on color images from the STL-10 dataset. These
% features will be used in an later exercise on convolution and pooling for 
% classifying STL-10 images.
clf; close all; clear all;

%%=========================================================================
%% STEP 0: INITIALIZE PARAMETERS
%  Here we initialize some parameters used for the exercise.

DEBUG = false;          % Set to TRUE for gradient checking
imageChannels = 3;     % number of channels (rgb, so 3)
patchDim   = 8;          % patch dimension
numPatches = 100000;   % number of patches
visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 
sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term       
epsilon = 0.1;	       % epsilon for ZCA whitening

%%=========================================================================
%% STEP 1: CHECK GRADIENTS

% To speed up gradient checking, we will use a reduced network and some
% dummy patches

if DEBUG
    debugHiddenSize = 5;
    debugvisibleSize = 8;
    patches = rand([8 10]);
    theta = initializeParameters(debugHiddenSize, debugvisibleSize); 

    % Get analytic gradient
    [cost, grad] = sparseAutoencoderLinearCost(theta, debugvisibleSize, ...
        debugHiddenSize, lambda, sparsityParam, beta, patches);

    % Get numerical gradient
    numGrad = computeNumericalGradient( @(x) ...
        sparseAutoencoderLinearCost(x, debugvisibleSize,...
        debugHiddenSize, lambda, sparsityParam, beta, patches), theta);

    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 

    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. Difference is usually less than 1e-9.
    disp(diff); 

    assert(diff < 1e-9, ...
        'Difference too large. Check your gradient computation again');
end

%%=========================================================================
%% STEP 2: LEARN FEATURES ON SMALL PATCHES
%  In this step, you will use your sparse autoencoder (which now uses a 
%  linear decoder) to learn features on small patches sampled from related
%  images.

%% STEP 2a: Load patches
%  In this step, we load 100k patches sampled from the STL10 dataset and
%  visualize them. Note that these patches have been scaled to [0,1]

load stlSampledPatches.mat

displayColorNetwork(patches(:, 1:100));

%% STEP 2b: Apply preprocessing
%  In this sub-step, we preprocess the sampled patches, in particular, 
%  ZCA whitening them. 
% 
%  In a later exercise on convolution and pooling, you will need to  
%  replicate exactly the preprocessing steps you apply to these patches  
%  before using the autoencoder to learn features on them. Hence, we will 
%  save the ZCA whitening and mean image matrices together with the learned 
%  features later on.

% Subtract mean patch (hence zeroing the mean of the patches)
meanPatch = mean(patches, 2);  
patches = bsxfun(@minus, patches, meanPatch);

% Apply ZCA whitening
sigma = patches * patches' / numPatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;

displayColorNetwork(patches(:, 1:100));

%% STEP 2c: Learn features
%  You will now use your sparse autoencoder (with linear decoder) to learn
%  features on the preprocessed patches. This should take ~45 minutes.

theta = initializeParameters(hiddenSize, visibleSize);

% Use minFunc to minimize the function
addpath minFunc/

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = 400;
options.display = 'on';

[optTheta, cost] = minFunc( @(p) sparseAutoencoderLinearCost(p, ...
    visibleSize, hiddenSize, lambda, sparsityParam, beta, patches), ...
    theta, options);

% Save the learned features and the preprocessing matrices for use in 
% the later exercise on convolution and pooling
fprintf('Saving learned features and preprocessing matrices...\n');                          
save('STL10Features.mat', 'optTheta', 'ZCAWhite', 'meanPatch');
fprintf('Saved\n');

%% STEP 2d: Visualize learned features

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+...
    hiddenSize);
displayColorNetwork( (W*ZCAWhite)');