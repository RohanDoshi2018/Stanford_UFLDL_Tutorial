%% Stanford UFLDL Tutorial (CS294) Ch5 Exercise
% This program implements a deep learning neural network with fine tuning
% for the task of digit recignition. There are four layers: an input layer,
% two hidden layers (hence two autoencoders stacked on top of each other
% for learning features), and a softmax classifier.
clf; close all; clear all;

%%=========================================================================
%% STEP 0: DECLARE PARAMETERS
inputSize = 28 * 28;     % # of input nodes
hiddenSizeL1 = 200;      % Layer 1 Hidden Size
hiddenSizeL2 = 200;      % Layer 2 Hidden Size
numClasses = 10;         % number of output labels
sparsityParam = 0.1;     % desired average activation of the hidden units.
lambda = 3e-3;           % weight decay parameter   
lambdaClassifier = 1e-4; % classifier lambda
beta = 3;                % weight of sparsity penalty term       
AE1maxIter = 400;        % max # iterations for autoencoder
AE2maxIter = 400;        % max # iterations for autoencoder
classifierMaxIter = 100; % max # iterations for softmax classifer
finetuneMaxIter = 400;   % max # iterations for finetuning

%%=========================================================================
%% STEP 1: LOAD DATA
%
% This loads our training data from the MNIST database files.

% Load MNIST database files: 60,000 samples
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

% Remap label 0 to 10
trainLabels(trainLabels == 0) = 10; 

%%=========================================================================
%% STEP 2: Train the first sparse autoencoder
% Trian the first sparse autoencoder on the unlabelled STL training
% images.

% Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Choose optimization method
options.display = 'on';
options.maxIter = AE1maxIter;

[SAE1opttheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
    inputSize, hiddenSizeL1, lambda, sparsityParam, beta, trainData), ...
    sae1Theta, options);    

%%=========================================================================
%% STEP 2: TRAIN THE SPRASE AUTOENCODER
% This trains the second sparse autoencoder on the first autoencoder
% features.

sae1Features = feedForwardAutoencoder(SAE1opttheta, hiddenSizeL1, ...
                                        inputSize, trainData);

% Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

options.maxIter = AE2maxIter;

[SAE2opttheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
    hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, ...
    sae1Features), sae2Theta, options);

%%=========================================================================
%% STEP 3: TRAIN THE SOFTMAX CLASSIFIER
%  This trains the sparse autoencoder on the second autoencoder features.

sae2Features = feedForwardAutoencoder(SAE2opttheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

options.maxIter = classifierMaxIter;

[saeSoftmaxOptTheta, ~] = minFunc( @(p) softmaxCost(p, numClasses, ...
    hiddenSizeL2, lambdaClassifier, sae2Features, trainLabels), ...
    saeSoftmaxTheta, options);

% reshape parameters into vector
saeSoftmaxOptTheta = saeSoftmaxOptTheta(:);

%%=========================================================================
%% STEP 4: FINETUNE THE MODEL
% Fine tune the model, treating it as one large neural network.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(SAE1opttheta(1:hiddenSizeL1*inputSize), ...
    hiddenSizeL1, inputSize);
stack{1}.b = SAE1opttheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1* ...
    inputSize+hiddenSizeL1);
stack{2}.w = reshape(SAE2opttheta(1:hiddenSizeL2*hiddenSizeL1), ...
    hiddenSizeL2, hiddenSizeL1);
stack{2}.b = SAE2opttheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2* ...
    hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

options.maxIter = finetuneMaxIter;
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
    inputSize, hiddenSizeL2, numClasses, netconfig, lambda, trainData, ...
    trainLabels), stackedAETheta, options);   
                               
%%=========================================================================
%% STEP 5: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images: 10000 samples
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

% Remap 0 to 10
testLabels(testLabels == 0) = 10; 

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
%
% Results: 400 iterations for AE1, AE2, finetuning
%          100 iterations for softmax classifier
%
% Stanford:
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% My Implementation:
% Before Finetuning Test Accuracy: 91.930%
% After Finetuning Test Accuracy:  97.820% 