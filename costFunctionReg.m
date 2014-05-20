function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

one = zeros(m,1);
n = length(X(1,:));



thetaSq = 0;

for j = 2:n  % You should not vectorize theta0, that's why it starts from 2  
    thetaSq = thetaSq + theta(j,1)^2;  
end
  
disp("thetaSq is ...");
disp(thetaSq);

hypo = sigmoid(X * theta);


J = 1/m * ( transpose(-y) * log(hypo) - transpose(1 - y) * log(1 - hypo) ) + (lambda / (2*m)) * thetaSq;
 
disp("(lambda / 2*m) * thetaSq is...");
disp((lambda / (2*m)) * thetaSq);
disp("Cost is ...");
disp(J);
 


grad = 1/m * ( transpose(X) * (hypo - y) ) + (lambda / m) * theta;
gradTheta0 = 1/m * ( transpose(X) * (hypo - y) );
grad(1,1) = gradTheta0(1,1); 


% since we are using a built-in function fminunc in octave to find the optimal parameters for unconstrained function, so we only need to calculate the initial theta

disp("grad...");
disp(grad);

% =============================================================

end
