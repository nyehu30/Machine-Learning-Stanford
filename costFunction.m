function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% theta = n x 1
% X = m x n
% y = m x 1




hypo = sigmoid(X * theta); % m x n * n x 1 = m x 1, this part should be correct

J = 1 / m * ( transpose(-y) * log(hypo) - transpose(1 - y) * log(1 - hypo) ); % we get an answer of 1 

%for i = 1:m
%    J = J + 1/m * ( -y(i,1) * log(hypo(i,1)) - (1 - y(i,1))*log(1 - hypo(i,1)) );
%end

grad = 1/m * (transpose(X) * (hypo - y)); % since we are using a built-in function fminunc in octave to find the optimal parameters for unconstrained function, so we only need to calculate the initial theta
disp(grad);
% =============================================================

end
