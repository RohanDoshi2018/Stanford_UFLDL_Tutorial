function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
[nfeatures nsamples] = size(data);
groundTruth = full(sparse(labels, 1:nsamples, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

depth = numel(stack);
a = cell(depth + 1, 1);
a{1} = data;
for layer = 1 : depth
    a{layer + 1} = bsxfun(@plus, stack{layer}.w * a{layer}, stack{layer}.b);
    a{layer + 1} = sigmoid(a{layer + 1});
end
M = softmaxTheta * a{depth + 1};
M = bsxfun(@minus, M, max(M));
p = bsxfun(@rdivide, exp(M), sum(exp(M)));
cost = - sum(sum(groundTruth .* log(p))) ./ nsamples;
cost = cost + sum(sum(softmaxTheta .^ 2)) .* lambda ./ 2;  % apply complexity cost here
% this part is confusing
temp = (groundTruth - p) * a{depth+1}';
temp = - temp ./ nsamples;
softmaxThetaGrad = temp + lambda .* softmaxTheta; % apply complexity cost here
delta = cell(depth + 1);
delta{depth+1} = -(softmaxTheta' * (groundTruth - p)) .* dsigmoid(a{depth+1});
for layer = depth : -1 : 2 
delta{layer} = (stack{layer}.w' * delta{layer+1}) .* dsigmoid(a{layer});
end
for layer = depth : -1 : 1 % weight penalty is missing from lower terms... % apply complexity cost here
stackgrad{layer}.w = delta{layer+1} * a{layer}';
stackgrad{layer}.b = sum(delta{layer+1}, 2);
stackgrad{layer}.w = stackgrad{layer}.w ./ nsamples;
stackgrad{layer}.b = stackgrad{layer}.b ./ nsamples;
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

%-------------------------------------------------------------------
% This function calculate dSigmoid
%
function dsigm = dsigmoid(a)
    dsigm = a .* (1.0 - a);
end