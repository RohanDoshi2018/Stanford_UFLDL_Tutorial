function theta = initializeParameters(hiddenSize, visibleSize)
% This function initializes weights and biases for an 
% autoencoder  and outputs these parameters as a flattened vector, as
% required by minFunc optimization library.
%
% hiddenSize: # of nodes in hidden layer
% visableSize: $ of nodes in visable layer and output layer

% initialization values are random but based on layer size
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1); 

% Weights are chosen uniformly on interval from [-r,r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

% biases set to zero initially
b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

% flatten all weight and bias matrices into a single vector 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
end