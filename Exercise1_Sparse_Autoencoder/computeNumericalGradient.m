function numgrad = computeNumericalGradient(J, theta)
% Compute the numerical gradient of a function (J) using finite difference 
% approximations at a given input (theta).

% initialize parameters
numgrad = zeros(size(theta));
epsilon = 10^-4;

for i=1:length(theta)
    % command line counter
    if (mod(i, 100) == 0) 
        fprintf(' %d', i);
        if (mod(i, 2000) == 0)
            fprintf('\n');
        end
    end
    
    % calcualate numerical gradient    
    e = zeros(length(theta),1);
    e(i) = 1;
    thetaPlus = theta + epsilon * e;
    thetaMinus = theta - epsilon * e;
    numgrad(i) = (J(thetaPlus) - J(thetaMinus)) / (2 * epsilon);
end

end