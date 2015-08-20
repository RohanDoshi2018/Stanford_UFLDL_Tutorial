%% Stanford UFLDL Tutorial (CS294) Ch1 Exercise
% This program trains a sparse autoencoder on sampled image patches.
% It features a gradient checking safeguard to ensure the cost function
% calld by the optimiation function each iteration provides the corrct 
% cost and gradient.
clf; close all; clear all;

%%=========================================================================
%% STEP 0: SELECT PARAMETERS
visibleSize = 8*8;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.01;% desired average activation of the hidden units.
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term     
nPatches = 10000;    % number of patches to be taken from sample images
DEBUG = false;       % debug mode checks the gradient of the cost function

%%=========================================================================
%% STEP 1: GATHER IMAGE PATCHES
% Data will be retrieved using sampleImages.m. Then, 200 random
% patches will be displayed for viewing.
patches = sampleIMAGES(sqrt(visibleSize),nPatches);
display_network(patches(:,randi(size(patches,2),200,1)),8);

% Randomly initiaize weight and bias parameters
theta = initializeParameters(hiddenSize, visibleSize);

%%=========================================================================
%% STEP 2: GRADIENT CHECKING
% Confirm that sparseAutoencoderCost.m provides the same gradient 
% analytically (using back-propagation) and numerically (using
% the finite difference approximation).
if DEBUG 
    % Get the analytic gradient (grad) using sparseAutoencoderCost.m as a 
    % reference.
    [cost, grad] = sparseAutoencoderCost(theta, visibleSize, ...
        hiddenSize, lambda, sparsityParam, beta, patches);
    
    numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, ...
        visibleSize, hiddenSize, lambda, sparsityParam, beta, patches), ...
        theta);
    
    % Compare the gradients side by side
    disp([numgrad grad]); 

    % Quantize the the analytic and numerical gradient difference
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % Should be small. In our implementation, these values are
                % usually less than 1e-9.
end

%%=========================================================================
%% STEP 3: TRAIN THE SPARSE AUTOENCODER
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Choose optimization method
options.maxIter = 400;
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
    visibleSize, hiddenSize, lambda, sparsityParam, beta, patches), ...
    theta, options);

%%=========================================================================
%% STEP 4: VISUALIZATION
% Visualize the weights responsible for forming the hidden layer
% in the sparse autoencoder.
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 
print -djpeg weights.jpg   % save the visualization to a file 