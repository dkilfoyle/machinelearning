library(ggplot2)
library(dplyr)

load_ex5 = function() {
  library(R.matlab)
  return(readMat("ex5/ex5data1.mat"))
}

list2env(load_ex5(),.GlobalEnv)

# data.frame(X=X,y=y) %>% 
#   ggplot(aes(X,y)) + geom_line()

m = nrow(X)

addBias = function(X) {
  return(cbind(rep(1, nrow(X)), X))
}

linearRegCostFunction = function(theta, X, y, lambda=0) {
  # X should have a column of 1s at the start for bias
  m = length(y)
  theta = as.vector(theta)
  ho = X %*% theta
  J = (1/(2*m) * sum((ho-y)^2))
  theta[1] = 0 # dont't penalty the bias theta
  J = J + (lambda / (2 * m)) * sum(theta^2)
  
  return(J)
}


linearRegGradFunction = function(theta, X, y, lambda=0) {
  # X should have a column of 1s at the start for bias
  theta = as.vector(theta)
  m=length(y)
  gradient = 1/m * t(X) %*% ((X %*% theta) - y)
  theta[1] = 0 # don't include penalty term for bias theta
  gradient = gradient + (lambda / m) * theta
  return(gradient)
}

trainLinearReg = function(X, y, lambda) {
  initial_theta = matrix(0, ncol(X), 1) 
  mylr=optim(initial_theta, linearRegCostFunction, linearRegGradFunction, X, y, lambda, method="BFGS")
  return(mylr$par)
}

predict.lr = function(theta, X) {
   addBias(X) %*% theta
}

# theta = cbind(1,1)
# J = linearRegCostFunction(theta, addBias(X), y, 1)
# grad = linearRegGradFunction(theta, addBias(X), y, 1)
# 
# lambda = 0
# theta = trainLinearReg(addBias(X), y, lambda)
# 
# data.frame(X=X,y=y) %>%
#   ggplot(aes(X,y)) + geom_line() +
#   geom_line(data=data.frame(X=X,y=predict.lr(theta,X)))

learningCurve = function(X, y, Xval, yval, lambda) {
  m = nrow(X)
  error_train = matrix(0, nrow=m, ncol=1)
  error_val   = matrix(0, nrow=m, ncol=1)
  for (i in 1:m) {
    # train the thetas using 1:i examples
    Xi = matrix(X[1:i,],nrow=i)
    yi = matrix(y[1:i,],nrow=i)
    thetas = trainLinearReg(Xi, yi, lambda)
    
    error_train[i] = linearRegCostFunction(thetas, Xi, yi, lambda=0)
    error_val[i] = linearRegCostFunction(thetas, Xval, yval, lambda=0)
  }
  return(list(error_train=error_train, error_val=error_val))
}

lc = learningCurve(addBias(X), y, addBias(Xval), yval, 0)

data.frame(x=1:m, train=lc$error_train, validate=lc$error_val) %>% 
  ggplot(aes(x=x)) +
    geom_line(aes(y=train)) +
    geom_line(aes(y=validate))
