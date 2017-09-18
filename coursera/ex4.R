library(ggplot2)
library(latex2exp)
library(shiny)
library(dplyr)

g = function(z) { 1 / (1 + exp(-z)) }

dgdz = function (z) {
  return (g(z)*(1-g(z)))
}

nnCost = function(thetas, input_layer_size, hidden_layer_size, num_labels, X, y, lambda=0) {
  
  # Reshape the unrolled thetas into the weight matrices
  theta1_size = hidden_layer_size * (input_layer_size + 1)
  theta1 = matrix(thetas[1:theta1_size], nrow=hidden_layer_size, ncol=(input_layer_size+1))
  theta2 = matrix(thetas[(theta1_size+1):length(thetas)], nrow=num_labels, ncol=(hidden_layer_size+1))
  
  # Setup some useful variables
  m = nrow(X)
  n = ncol(X)
  
  J = 0
  Delta_l1 = matrix(0, nrow=nrow(theta1), ncol=ncol(theta1))
  Delta_l2 = matrix(0, nrow=nrow(theta2), ncol=ncol(theta2))
  
  for (i in 1:m) { 
    y_k = diag(num_labels)[y[i],]   # eg y=3 = c(0,0,1,0,0,0,0,0,0,0)
    
    a1 = rbind(1, X[i, ]) # a(1) = X(1) with an extra 1 for a_0
    a2 = g(theta1 %*% a1) # input to hidden
    a2 = rbind(1, a2) # add a_0^(2)
    h = g(theta2 %*% a2) # hidden to output
    a3 = h
    
    # vectorise the sum over k labels
    cost = 1/m * (t(-y_k) %*% log(h) - (t(1-y_k) %*% log(1-h)))
    J = J + cost
    
    delta_L = a3 - y_k
    delta_l3 = L
    delta_l2 = (t(theta2) %*% delta_l3) * a2 * (1-a2)
    
    Delta_l2 = Delta_l2 + (delta_l3 %*% t(a2))
    Delta_l1 = Delta_l1 + (delta_l2 %*% t(a1))
    
  }
  
  penalty = (lambda / (2 * m)) * (sum(theta1[,-1]^2) + sum(theta2[,-1]^2))
  J = J + penalty
  
  # average the Delta and add regularisation
  theta1_0bias = theta1
  theta1_0bias[,1] = rep(0, nrow(theta1))
  theta2_0bias = theta2
  theta2_0bias[,1] = rep(0, nrow(theta2))
  D1 = 1/m * Delta_l1 + lambda * theta1_0bias # exclude the bias column in theta
  D2 = 1/m * Delta_l2 + lambda * theta2_0bias
  
  return(list(J=J,D1=D1,D2=D2))
}

randInitializeWeights = function(in_layer_size, out_layer_size) {
  # Randomly initialize the weights to small values
  epsilon_init = 0.12
  # epsilon_init = sqrt(6) / (sqrt(in_layer_size + out_layer_size)) # recommended strategy
  matrix(runif(out_layer_size * (in_layer_size+1)), nrow=out_layer_size) * 2 * epsilon_init - epsilon_init
}

debugInitializeWeights = function(fan_out, fan_in) {
  matrix(sin(1:((fan_in+1)*fan_out))/10.0, 
    nrow=fan_out, 
    ncol=(fan_in+1))
}

checkNNGradients = function(lambda=0) {

  input_layer_size = 3
  hidden_layer_size = 5
  num_labels = 3
  m = 5

  # generate some 'random' test data
  Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
  Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
  
  # Reusing debugInitializeWeights to generate X
  X  = debugInitializeWeights(m, input_layer_size - 1)
  y  = cbind(1 + (1:m %% num_labels))

  # Unroll parameters
  nn_params = c(as.vector(Theta1), as.vector(Theta2))

# % Short hand for cost function
# costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
# num_labels, X, y, lambda);
# 
# [cost, grad] = costFunc(nn_params);
# numgrad = computeNumericalGradient(costFunc, nn_params);
# 
# % Visually examine the two gradient computations.  The two columns
# % you get should be very similar. 
# disp([numgrad grad]);
# fprintf(['The above two columns you get should be very similar.\n' ...
# '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
# 
# % Evaluate the norm of the difference between two solutions.  
# % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# % in computeNumericalGradient.m, then diff below should be less than 1e-9
# diff = norm(numgrad-grad)/norm(numgrad+grad);
# 
# fprintf(['If your backpropagation implementation is correct, then \n' ...
# 'the relative difference will be small (less than 1e-9). \n' ...
# '\nRelative Difference: %g\n'], diff);

}


loadex4 = function() {
  library(R.matlab)
  ex4data1 <<- readMat("ex4/ex4data1.mat")
  ex4weights <<- readMat("ex4/ex4weights.mat")
}

# loadex4()
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10
nn_params = c(as.vector(ex4weights$Theta1), as.vector(ex4weights$Theta2))

# nnCost(nn_params, input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y)
# nnCost(nn_params, input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y, lambda=1)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = c(as.vector(initial_Theta1), as.vector(initial_Theta2))


