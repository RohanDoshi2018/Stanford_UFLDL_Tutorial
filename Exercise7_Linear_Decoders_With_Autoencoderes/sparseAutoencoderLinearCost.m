function [cost, grad] = sparseAutoencoderLinearCost(theta, ...
    visibleSize, hiddenSize, lambda, sparsityParam, beta, data)
 
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  
%       So, data(:,i) is the i-th training example. 
  
%% Prepare variables

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), ...
    visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

numpatches = length(data); % probably 10,000

%% Forward Porpogation For Cost
a1 = data;
z2 = W1*data+repmat(b1,1,numpatches);
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,numpatches);
a3 = z3;

%average activation of a2 (average over all patches)
pj = sum(a2, 2) ./ numpatches; 

% cost calculation

SquareError = sum(sum(.5 * (data-a3) .* (data-a3))) ./ numpatches;
WeightDecayError = (lambda/2) *(sum(sum(W1 .* W1)) + sum(sum(W2 .* W2)));
SparsityPenaltyTerm = sparsityParam * log(sparsityParam ./ pj) + ...
    (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - pj));
SparsityPenaltyTerm = beta * sum(SparsityPenaltyTerm);
cost = SquareError + WeightDecayError + SparsityPenaltyTerm;

%% Calculate Gradient

% Back Propagate Error (a.k.a "delta")
delta3 = -(data-a3);
delta2SparsityTerm = beta .* ( -sparsityParam ./pj + ...
    (1-sparsityParam) ./(1-pj));
delta2 = (W2'*delta3 + repmat(delta2SparsityTerm,1,numpatches) ) .* ...
    sigmoidPrime(z2);

W2grad = (W2grad + delta3 * a2') ./ numpatches + lambda .* W2;
W1grad = (W1grad + delta2 * a1') ./ numpatches + lambda .* W1;
b2grad = sum(delta3, 2) ./ numpatches;
b1grad = sum(delta2, 2)./ numpatches;

% Roll the gradient
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

% Sigmoid
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Derivative of Sigmoid
function y = sigmoidPrime(x)
    y = sigmoid(x) .* (1-sigmoid(x));
end
