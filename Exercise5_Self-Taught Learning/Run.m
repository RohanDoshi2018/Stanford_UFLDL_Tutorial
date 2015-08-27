%% Stanford UFLDL Tutorial (CS294) Ch5 Exercise
% This program trains the features extracted from a sparse autoencoder
% with a softmax classifier. The overall task is to identify handwritten
% digits.
clf; close all; clear all;

%% ========================================================================
%%  STEP 0: DECLARE PARAMETERS
inputSize  = 28 * 28;     % # of input nodes
numLabels  = 5;           % number of output labels
hiddenSize = 200;         % hidden size of autoencoder
sparsityParam = 0.1;      % desired average activation of the hidden units
lambdaAE = 3e-3;          % weight decay parameter for autoencoder
lambdaClassifier = 1e-4;  % weight decay for softmax classifier
beta = 3;                 % weight of sparsity penalty term
AEmaxIter = 400;          % max # iterations for autoencoder
classifierMaxIter = 100;  % max # iterations for softmax classifer

%% ========================================================================
%%  STEP 1: LOAD DATA
% Load training and testing data from the MNIST database files. Then, 
% split training data into labeled data (for training the softmax
% classifer) and unlabelled data (for training the autoencoder).

% Load MNIST database files
mnistData   = loadMNISTImages('mnist/train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

% Create a labeled set (0 to 4). Split labeled data into train and test
% set for softmax classifier

labeledSet = find(mnistLabels >= 0 & mnistLabels <= 4);

numTrain = round(numel(labeledSet)/2); 
trainSet = labeledSet(1:numTrain); % Set ==> array of indices
trainData = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testSet = labeledSet(numTrain+1:end); % Set ==> array of indices
testData = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Create an unlabeled set (5 to 9) for training the sparse autoencoder for
% feature recognition

unlabeledSet = find(mnistLabels >= 5);
unlabeledData = mnistData(:, unlabeledSet);

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n', size(testData, 2));

%% ========================================================================
%% STEP 2: TRAIN THE SPRASE AUTOENCODER
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % select optimization function
options.display = 'on';
options.maxIter = AEmaxIter;

[opttheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, inputSize, ...
    hiddenSize, lambdaAE, sparsityParam, beta, unlabeledData), theta, ...
    options);                                 
                                 
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: EXTRACT FEATURES FROM SUPERVISED LABELED DATASET
% Use weights learned from unlabeled data to forward propogate
% labeled data for feature extraction.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
    trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
	testData);

%%======================================================================
%% STEP 4: TRAIN SOFTMAX CLASSIFIER
% Use the labeled training data to train the classifier on the features
% from the unsupervised autoencoder.

% initialize parameters
theta = 0.005 * randn(numLabels * hiddenSize, 1);

options.maxIter = classifierMaxIter;

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, numLabels, ...
    hiddenSize, lambdaClassifier, trainFeatures, trainLabels), theta, ...                                   
    options);

% reshape parameters from vector into matrix
softmaxOptTheta = reshape(softmaxOptTheta, numLabels, hiddenSize);

%%======================================================================
%% STEP 5: TESTING
% Test trained softmax classifier on the extracted features from the test
% data. Then, calculate the prediction's accuracy.
M = softmaxOptTheta * testFeatures;
M = bsxfun(@minus, M, max(M, [], 1)); %prevents overflows
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));
[maxProb, indexOfMax] = max(M,[], 1); % Gets index of max prob in each col
pred = indexOfMax;
acc = mean(testLabels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% Accuracy is the proportion of correctly classified images
% The results for our implementation (400 iterations for autoencoder and 
% 100 iterations for softmax classifier) was:
% Accuracy = 98.3