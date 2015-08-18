%% Stanford UFLDL Tutorial (CS294) Ch4 Exercise
% This program applies a softmax classifer to MNIST image patches directly.
% It trains on labeled data and makes predictions during testing.
clf; close all; clear all;
%%=========================================================================
%% STEP 0: DECLARE PARAMETERS
inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)
lambda = 1e-4;       % Weight decay parameter
softmaxMaxIter = 100;% Max # of iterations to train classifier
DEBUG = false;       % if true, apply gradient checking
%%=========================================================================
%% STEP 1: LOAD DATA
% load training images and labels
trainImages = loadMNISTImages('/mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('/mnist/train-labels-idx1-ubyte');
trainLabels(trainLabels==0) = 10; % Remap 0 to 10

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);                               
%%=========================================================================
%% STEP 2: GRADIENT CHECKING
% Compare the analytical and numerical gradient and check if they are the
% same.                                
if DEBUG
    % Get analytic gradient
    [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, ...
        trainImages, trainLabels);
    % Get numerical gradient
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
        inputSize, lambda, trainImages, trainLabels), theta);
    % Compare the gradients side by side
    disp([numgrad grad]); 

    % Quantize the the analytic and numerical gradient difference
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % Should be small. In our implementation, these values are
                % usually less than 1e-9.
end
%%=========================================================================
%% STEP 3: SOFTMAX CLASSIFIER TRAINING
                                       
% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Choose optimization method
options.display = 'on';
options.maxIter = softmaxMaxIter;

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, numClasses, ...
    inputSize, lambda, trainImages, trainLabels), theta, options);                       

softmaxOptTheta = reshape(softmaxOptTheta, numClasses, inputSize);
%%=========================================================================
%% STEP 4: TESTING
% Test the trainined softmax model of 10k test samples.

% load test images and labels
testImages = loadMNISTImages('/mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('/mnist/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

M = softmaxOptTheta * testImages;
M = bsxfun(@minus, M, max(M, [], 1)); %prevents overflows
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));
[maxProb, indexOfMax] = max(M,[], 1); % Gets index of max prob in each col

pred = indexOfMax;

acc = mean(testLabels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% After 100 iterations, expected accuracy: 92.200%