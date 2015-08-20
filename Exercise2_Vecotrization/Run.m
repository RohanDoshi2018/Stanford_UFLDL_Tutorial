%% Stanford UFLDL Tutorial (CS294) Ch2 Exercise
% This is a vectorized version of a sparse autoencoder. We train on the 
% MNIST handwritten digit dataset and visualize feature extraction
% on top of the testing images.
clf; close all; clear all;

%%======================================================================
%% STEP 0: SELECT PARAMETERS
visibleSize = 28*28; % number of input units 
hiddenSize = 196;    % number of hidden units 
sparsityParam = 0.1; % desired average activation of the hidden units.
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term     
DEBUG = false;       % set DEBUG to true for gradient checking

%%======================================================================
%% STEP 1: PREPARE TRAINING IMAGES
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

%only select first 10000 digit images
patches = images(:,1:10000);

% view 100 random  patches and 10 random labels to confirm MNIST is loaded
display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

%  Obtain random parameters theta
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