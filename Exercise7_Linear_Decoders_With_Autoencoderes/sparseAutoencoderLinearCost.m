function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------        


% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 


numpatches = length(data); % probably 10,000

% forward propogation
a1 = data;
z2 = W1*data+repmat(b1,1,numpatches);
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,numpatches);
a3 = z3;

%average activation of a2 (average over all patches)
pj = sum(a2, 2) ./ numpatches; 

% cost calculation
% note: x * X = ||x||^2
% note: sum(matrix) ==> summates columns, then rows
SquareError = sum(sum(.5 * (data-a3) .* (data-a3))) ./ numpatches;

WeightDecayError = (lambda/2) *(sum(sum(W1 .* W1)) + sum(sum(W2 .* W2)));

SparsityPenaltyTerm = sparsityParam * log(sparsityParam ./ pj) + (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - pj));
SparsityPenaltyTerm = beta * sum(SparsityPenaltyTerm);

cost = SquareError + WeightDecayError + SparsityPenaltyTerm;
%backpropogation
% "W2grad" does not refer to the gradient of W2, but, rather, the W2 
% matrix whose parameters will be returned in the vararible "grad" at 
% the end of the function. "grad" becomes the "theta" variable in other
% programs that keeps track of all the paramteres. This program simply
% evaluates the current theta's cost (error) with forward propogation.
% Then, it conducts conducts back-propogation once to compute partial
% derivatives, which will update theta values to be one step closer to
% an ideal fit. This program is called by an optimization function over
% and over to update the "theta" values.
 
delta3 = -(data-a3);

delta2SparsityTerm = beta .* ( -sparsityParam ./pj + (1-sparsityParam) ./(1-pj));
delta2 = (W2'*delta3 + repmat(delta2SparsityTerm,1,numpatches) ) .* sigmoidPrime(z2);

W2grad = (W2grad + delta3 * a2') ./ numpatches + lambda .* W2;
W1grad = (W1grad + delta2 * a1') ./ numpatches + lambda .* W1;
b2grad = sum(delta3, 2) ./ numpatches;
b1grad = sum(delta2, 2)./ numpatches;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

%-------------------------------------------------------------------
% Here's an implementation of the derivative of the sigmoid function.
% Recall the identity, for sigmoid function "f", f'(x) = f(x)(1-f(x)).

function y = sigmoidPrime(x)
    y = (1 ./ (1 + exp(-x))) .* (1 - (1 ./ (1 + exp(-x))) );
end
