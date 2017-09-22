library(ggplot2)
library(latex2exp)
library(shiny)
library(dplyr)
library(lbfgsb3)

source("lbfgsb3_.R")
source("displayData.R")

zeros = function(dims) {
  matrix(0, nrow=dims[1], ncol=dims[2])
}

numel = function(m) {
  return(prod(dim(m)))
}

sigmoid = function(z) { 1 / (1 + exp(-z)) }

sigmoidGradient = function (z) {
  return (g(z)*(1-g(z)))
}

nnCostFunction = function(input_layer_size, hidden_layer_size, num_labels,X, y, lambda=0) {
  function(nn_params) {
    # Reshape the unrolled thetas into the weight matrices
    Theta1_size = hidden_layer_size * (input_layer_size + 1)
    Theta1 = matrix(nn_params[1:Theta1_size], nrow=hidden_layer_size, ncol=(input_layer_size+1))
    Theta2 = matrix(nn_params[(Theta1_size+1):length(nn_params)], nrow=num_labels, ncol=(hidden_layer_size+1))
    
    # m is the number of samples (rows) in X
    m <- nrow(X)
    
    # recode y [4 2 3 2 4 1 ...] to Y
    # Y[1,] = 0 0 0 1 0 
    # Y[2,] = 0 1 0 0 0
    # Y[m,] ....
    I <- diag(num_labels)
    Y <- matrix(0, m, num_labels)
    for (i in 1:m) { Y[i,] <- I[y[i],] }
    
    # l1=4=i              i,j theta       l2=3=j    m=6
    # 1 a11 a12 a13 a14   O10 O20 O30     z1 z2 z3  for i=1
    # 1 a21 a22 a23 a24   O11 O21 O31     z1 z2 z3  for i=2
    # 1 a31 a32 a33 a34   O12 O22 O32  =  z1 z2 z3  for i=3
    # 1 a41 a42 a43 a44   O13 O23 O33     z1 z2 z3  for i=4
    # 1 a51 a52 a53 a54   O14 O24 O34     z1 z2 z3  for i=5
    # 1 a61 a62 a63 a64                   z1 z2 z3  for i=6
    
    # feedforward - vectorized over i..m
    a1 = cbind(rep(1,m),X)
    z2 = a1 %*% t(Theta1)
    a2 = cbind(rep(1,dim(z2)[1]), sigmoid(z2))
    z3 = a2 %*% t(Theta2)
    a3 = sigmoid(z3)
    h  = a3
    
    # calculte regularisation penalty
    p = sum(Theta1[,-1] ^ 2) + sum(Theta2[,-1] ^ 2)
    
    # calculate Cost
    J = sum((-Y) * log(h) - (1 - Y) * log(1 - h)) / m + lambda * p / (2 * m)
    
    return(J)
  }
}

nnGradientFunction = function(input_layer_size, hidden_layer_size, num_labels, X, y, lambda) {
  function(nn_params) {
    # Reshape the unrolled thetas into the weight matrices
    Theta1_size = hidden_layer_size * (input_layer_size + 1)
    Theta1 = matrix(nn_params[1:Theta1_size], nrow=hidden_layer_size, ncol=(input_layer_size+1))
    Theta2 = matrix(nn_params[(Theta1_size+1):length(nn_params)], nrow=num_labels, ncol=(hidden_layer_size+1))
    
    # m is the number of samples (rows) in X
    m = nrow(X)
    
    # You need to return the following variables correctly
    Theta1_grad = matrix(0, nrow=nrow(Theta1), ncol=ncol(Theta1))
    Theta2_grad = matrix(0, nrow=nrow(Theta2), ncol=ncol(Theta2))
    
    # recode y [4 2 3 2 4 1 ...] to Y[i, ] = c(0,0,0,1)
    I <- diag(num_labels)
    Y <- matrix(0, m, num_labels)
    for (i in 1:m) { Y[i,] = I[y[i],] }
    
    # feedforward - vectorized over i..m (same as costfunction)
    a1 = cbind(rep(1,m),X)
    z2 = a1 %*% t(Theta1)
    a2 = cbind(rep(1,nrow(z2)), sigmoid(z2))
    z3 = a2 %*% t(Theta2)
    a3 = sigmoid(z3)
    h  = a3
    
    # calculate sigmas = the "error" of each activation unit in a layer
    sigma3 = h - Y
    
    # for each activation unit in the previous layer scale the error it radiated to in the next node by the corresponding weight
    # sigma3    Theta2 j,i                Error scaled theta from l-1 unit
    # s1 s2 s3  O10 O11 O12 O13 O14 o15   etf0 etf1 etf2 etf3 etf4 etf5    i=1
    # s1 s2 s3  O20 O21 O22 O23 o24 o25 = etf0 etf1 etf2 etf3 etf4 etf5    i=2
    # s1 s2 s3  O30 O31 O32 O33 o34 o35   ...                              i=3
    # s1 s2 s3
    # s1 s2 s3
    # ....
    sigma2 = (sigma3 %*% Theta2) * sigmoidGradient(cbind(rep(1,nrow(z2)),z2))
    sigma2 = sigma2[,-1] # remove a0
    
    # accumulate gradients
    # scale each output error by the incoming activation
    # s1 s1 s1 s1 s1 s1 s1 s1 ..     a20 a21 a22 a23 a24 a25 (i=1)   
    # s2 s2 s2 s2 s2 s2 s2 s2 ..     a20 a21 a22 a23 a24 a25 (i=2) = 
    # s3 s3 s3 s3 s3 s3 s3 s3 ..     ..  ..  ..  ..  ..  ..  (i=m)
    delta_2 = (t(sigma3) %*% a2)   
    delta_1 = (t(sigma2) %*% a1)

    
    # calculate regularized gradient
    p1 = (lambda / m) * cbind(rep(0,dim(Theta1)[1]), Theta1[,-1])
    p2 = (lambda / m) * cbind(rep(0,dim(Theta2)[1]), Theta2[,-1])
    Theta1_grad = delta_1 / m + p1
    Theta2_grad = delta_2 / m + p2
    
    # Unroll gradients
    grad = c(c(Theta1_grad), c(Theta2_grad))
    return(grad)
  }
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

computeNumericalGradient = function(thetas, input_layer_size, hidden_layer_size, num_labels, X, y, lambda=0) {
  numgrad = rep(0, length(thetas))
  perturb = rep(0, length(thetas))
  e = 1e-4
  for (p in 1:length(thetas)) {
    perturb[p] = e
    loss1 = nnCost(thetas - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)$J
    loss2 = nnCost(thetas + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)$J
    # Compute Numerical Gradient
    numgrad[p] = (loss2 - loss1) / (2*e)
    perturb[p] = 0
  }
  return(numgrad)
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
  nn_params = c(Theta1, Theta2)

 
  backProp = nnGradientFunction(input_layer_size, hidden_layer_size, num_labels, X, y, lambda)(nn_params)
  numgrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
  
  print(cbind(backProp,numgrad))
  
  grad_diff = norm(as.matrix(numgrad-backProp))/norm(as.matrix(numgrad+backProp))
  print(grad_diff)
}


loadex4 = function() {
  library(R.matlab)
  ex4data1 <<- readMat("ex4/ex4data1.mat")
  ex4weights <<- readMat("ex4/ex4weights.mat")
}

# loadex4()
# input_layer_size  = 400  # 20x20 Input Images of Digits
# hidden_layer_size = 25   # 25 hidden units
# num_labels = 10
# nn_params = c(ex4weights$Theta1, ex4weights$Theta2)

# system.time(nnCostFunctionForLoop(input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y)(nn_params))
# system.time(nnCostFunction(input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y)(nn_params))

# print(nnCostFunctionForLoop(input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y,lambda=1)(nn_params))
# print(nnCostFunction(input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y, lambda=1)(nn_params))
# print(nnGradientFunctionForLoop(input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y,lambda=1)(nn_params))
# print(nnGradientFunction(input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y, lambda=1)(nn_params))

# 
# print(nnCost(nn_params, input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y)$J)
# print(nnCost(nn_params, input_layer_size, hidden_layer_size, num_labels, ex4data1$X, ex4data1$y, lambda=1)$J)

# checkNNGradients()

loadex4()
X = ex4data1$X
y = ex4data1$y
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = c(initial_Theta1, initial_Theta2)

lambda =1 

costFunction = nnCostFunction(input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
gradFunction = nnGradientFunction(input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

opt = lbfgsb3_(initial_nn_params, fn= costFunction, gr=gradFunction, control = list(trace=1,maxit=50))
nn_params = opt$prm
cost = opt$f

Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],  hidden_layer_size, (input_layer_size + 1))
Theta2 <- matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],  num_labels, (hidden_layer_size + 1))

displayData(Theta1[, -1])

# pred <- predict(Theta1, Theta2, X)

