function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
% Theta1: 25 x 401
% Theta2: 10 x 26
%

% input layer
X = [ones(m, 1) X];  % 5000 x 401  
z2 = Theta1 * X';  % 25 x 401 * 401 x 5000
a2 = sigmoid(z2);  % 25 * 5000
% hidden layer
a2 = [ones(1, m);a2];  % 26 * 5000 new 'X'
z3 = Theta2 * a2;  % 10 x 26 * 26 x 5000
a3 = sigmoid(z3);  % 10 x 5000
% output layer
output =a3';  % 5000 x 10
[c,i] = max(output, [], 2);  % the p has the max probabilities among the ten classes
p = i;  






% =========================================================================


end
