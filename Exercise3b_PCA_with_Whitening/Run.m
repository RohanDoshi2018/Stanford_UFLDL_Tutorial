%% Stanford UFLDL Tutorial (CS294) Ch3b Exercise
% This program applies various pre-processing steps to some sample
% data such as PCA and whitening.
clf; close all; clear all;
%%=========================================================================
%% Step 0a: Load data
% Here we provide the code to load natural image data into x.
% x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds 
% to the raw image data from the kth 12x12 image patch sampled.
x = sampleIMAGESRAW();
figure('name','Raw images');
randsel = randi(size(x,2),200,1); % 200 random samples for visualization
display_network(x(:,randsel));
%%=========================================================================
%% Step 0b: Zero-mean the data (by row)
%  compute the mean pixel intensity value for each patch
avg = mean(x,2);
x = x-repmat(avg,1, size(x,2));
%%================================================================
%% Step 1a: Implement PCA to obtain xRot
% Implement PCA to obtain xRot, the matrix in which the data is expressed
% with respect to the eigenbasis of sigma, which is the matrix U.
sigma = x * x' / size(x,2);
[U, S, V] = svd(sigma);
xRot = U' * x;
%%=========================================================================
%% Step 1b: Check your implementation of PCA
% The covariance matrix for the data expressed with respect to the basis U
% should be a diagonal matrix with non-zero entries only along the main
% diagonal. We will verify this here by computing the covariance.  
% We want to visualize the covar matrix and see a straight diagonal 
% line (non-zero entries) against a blue background (zero entries).
covar = S;
figure('name','Visualisation of covariance matrix');
imagesc(covar);
%%=========================================================================
%% Step 2: Find k, the number of components to retain
% Here, we detetermine k, the number of components to retain in order
% to retain at least 99% of the variance.
k = size(covar,1);
eigenValues = sum(covar,2); % turn covar into column vector
sumDiag = sum(eigenValues);
accuracy = 1;
while accuracy > .90
    k=k-1;
    accuracy = sum(eigenValues(1:k,1)) / sumDiag;
end
%%=========================================================================
%% Step 3: Implement PCA with dimension reduction
% Now that we have found k, we can reduce the dimension of the data by
% discarding the remaining dimensions. In this way, we can represent the
% data in k dimensions instead of the original 144, which will save 
% computational time when running learning algorithms on the reduced
% representation.
% 
% Following the dimension reduction, we invert the PCA transformation to  
% produce the matrix xHat, the dimension-reduced data with respect to the 
% original basis. We visualise the data and compare it to the raw data. 
% There will be littel loss loss since we are throwing away principal 
% components that coorespond to dimensions with lowest variation.

xRotReduced = xRot;
xRotReduced(k+1:size(xRot,1),:) = 0; % reduced dimension representation of  
                                     % data; k is # of eigenvalues kept
xHat = U * xRotReduced;
                     

% Visualise the data, and compare it to the raw data
% We should observe that the raw and processed data are of similar quality.

figure('name','Raw images');
display_network(x(:,randsel));
figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', ...
    k, size(x, 1)),'']);
display_network(xHat(:,randsel));
%%=========================================================================
%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 
epsilon = 0.1;
xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * xRot;
%%=========================================================================
%% Step 4b: Check your implementation of PCA whitening 
% We check PCA whitening with & without regularisation. 
% PCA whitening without regularisation results a covariance matrix 
% that is equal to the identity matrix. PCA whitening with regularisation
% results in a covariance matrix with diagonal entries starting close to 
% 1 and gradually becoming smaller. We will verify these properties here.
%
% Without regularisation (set epsilon to 0 or close to 0), 
% when visualised as an image, you should see a red line across the
% diagonal (one entries) .a red line that slowly turns
% blue across the diagonal, corresponding to the one entries slowly
% becoming smaller.
figure('name','Visualisation of covariance matrix 2');
imagesc(covar);
%%=========================================================================
%% Step 5: Implement ZCA whitening
% Now we implement ZCA whitening to produce the matrix xZCAWhite. 
% Visualise the data and compare it to the raw data. We should observe
% that whitening results in, among other things, enhanced edges.
xZCAWhite = U * xPCAWhite;
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));