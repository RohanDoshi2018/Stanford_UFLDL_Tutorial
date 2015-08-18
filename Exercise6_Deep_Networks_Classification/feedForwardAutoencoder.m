function [a2] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)

% theta: trained weights from the autoencoder
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
% reshape parameters from vector to block format
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

numpatches = size(data,2); % number of samples

% forward propogation
a1 = data;
z2 = W1*a1+repmat(b1,1,numpatches);
a2 = sigmoid(z2);
end

% Signmoid Function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
