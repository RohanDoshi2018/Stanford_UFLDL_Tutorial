function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, ...
    hiddenSize, lambda, sparsityParam, beta, data)
% Compute the cost (aka optimization objective) of J_sparse(W,b) to train
% the sparse autoencoder. Then, generate the gradients to update weights(W)
% and biases(b) contained in the "theta" input.
%
% Stated differently, if we were using batch gradient descent to optimize 
% parameters, the gradient descent update is W1 := W1 - alpha * W1grad
% for W1, with similar updates for W2, b1, b2. 
%
% theta: vectorized version of input parameters to optimize (W,b)
% visibleSize: the number of input units 
% hiddenSize: the number of hidden units 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units
% beta: weight of sparsity penalty term
% data: Matrix containing the training data. data(:,i) is the i-th 
%   training example. 
  
% turn theta back into matrix format
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), ...
    visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Initialize parameters
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% # sample patches
numpatches = length(data); 

% perform forward propogation
a1 = data;
z2 = W1*a1+repmat(b1,1,numpatches);
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,numpatches);
a3 = sigmoid(z3);

% calculate average activation of a2 (average over all patches)
pj = sum(a2, 2) ./ numpatches; 

% calculate cost
SquareError = sum(sum(.5 * (data-a3) .* (data-a3))) ./ numpatches;
WeightDecayTerm = (lambda/2) *(sum(sum(W1 .* W1)) + sum(sum(W2 .* W2)));
SparsityPenaltyTerm = sparsityParam * log(sparsityParam ./ pj) + ...
    (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - pj));
SparsityPenaltyTerm = beta * sum(SparsityPenaltyTerm);
cost = SquareError + WeightDecayTerm + SparsityPenaltyTerm;

% back-propagation of the error (a.k.a 'delta')
delta3 = -(data-a3) .* sigmoidPrime(z3);
delta2SparsityTerm = beta .* (-sparsityParam ./pj + (1-sparsityParam) ...
    ./(1-pj));
delta2 = (W2'*delta3 + repmat(delta2SparsityTerm,1,numpatches) ) .* ...
    sigmoidPrime(z2);

% calculate analytic gradient
W2grad = (W2grad + delta3 * a2') ./ numpatches + lambda .* W2;
W1grad = (W1grad + delta2 * a1') ./ numpatches + lambda .* W1;
b2grad = sum(delta3, 2) ./ numpatches;
b1grad = sum(delta2, 2)./ numpatches;

% flatten all gradients matrices into a single vector 
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

% Sigmoid
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Derivative of Sigmoid
% For sigmoid function "f", f'(x) = f(x)(1-f(x)).
function y = sigmoidPrime(x)
    y = sigmoid(x) .* (1-sigmoid(x));
end