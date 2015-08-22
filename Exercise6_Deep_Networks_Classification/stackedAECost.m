function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta & labeled training data,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning. 
%
% Note: We use the "stack" data structure to treat the model as one large
% neural network. But, the stack only includes the input and hidden layers,
% not the softmax classifier. 
%
% For example, if our model in need of fine tuning has one input layer, two 
% stacked autoencoders, and an output softmax classifier, the "stack"
% data structure only includes only the first three layers, and the "depth" 
% of the stack would be 2, representing the number of stacked autoencoders.
%
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  
%       So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

% Note that the softmax classifier is not considered as part of stack since
% it is not a true layer. For these reasons, calculations for the softmax
% parameters are kept seperate from the rest of the neural network.

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

% feedforward data to get probability matrix
depth = numel(stack); % depth = # of stacked autoencoders (hidden layers)
a = cell(depth + 1, 1);
a{1} = data;
for layer = 1 : depth
    a{layer + 1} = bsxfun(@plus, stack{layer}.w * a{layer}, stack{layer}.b);
    a{layer + 1} = sigmoid(a{layer + 1});
end
M = softmaxTheta * a{depth + 1};
M = bsxfun(@minus, M, max(M));
p = bsxfun(@rdivide, exp(M), sum(exp(M)));
p = log(p);

% find the cost
cost = - sum(sum(groundTruth .* p)) ./ nsamples;
cost = cost + sum(sum(softmaxTheta .^ 2)) .* lambda ./ 2;  % apply complexity cost here

% back-propagate to calculate error (a.k.a 'delta')
delta = cell(depth + 1);
delta{depth+1} = -(softmaxTheta' * (groundTruth - p)) .* ... % for softmax
    dsigmoid(a{depth+1}); 
for layer = depth:-1:2 % for stack
    delta{layer} = (stack{layer}.w' * delta{layer+1}) .* ...
        dsigmoid(a{layer}); 
end

% calculate the gradient for the softmax classifier
softmaxThetaGrad = -(groundTruth - p) * a{depth+1}' ./ nsamples + ... % apply complexity cost here
    lambda .* softmaxTheta; % for softmax %fix this lambda...

% calculate the gradient for all layers
for layer = depth:-1:1 % weight penalty is missing from lower terms... % apply complexity cost here
    stackgrad{layer}.w = delta{layer+1} * a{layer}' ./ nsamples;
    stackgrad{layer}.b = sum(delta{layer+1}, 2) ./ nsamples ;
end

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

% Sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

% Derivative of Sigmoid 
% Note: 'a' already has sigmoid applied to it e.g. a = f(z)
function dsigm = dsigmoid(a)
    dsigm = a .* (1.0 - a);
end