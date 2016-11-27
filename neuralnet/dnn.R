# sources:
# www.parallelR.com
# http://cs231n.github.io/neural-networks-case-study/

library(ggplot2)
library(tidyr)

# Prediction
predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad
  hidden.layer <- sweep(new.data %*% model$W1 ,2, model$b1, '+')
  # neurons : Rectified Linear
  hidden.layer <- pmax(hidden.layer, 0)
  score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
  
  # Loss Function: softmax
  score.exp <- exp(score)
  probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
  
  # select max possiblity
  labels.predicted <- max.col(probs)
  return(labels.predicted)
}

# Train: build and train a 2-layers neural network 
train.dnn <- function(x, y, traindata=data, testdata=NULL,
  model = NULL,
  # set hidden layers and neurons
  # currently, only support 1 hidden layer
  hidden=c(6), 
  # max iteration steps
  maxit=2000,
  # delta loss 
  abstol=1e-2,
  # learning rate
  lr = 1e-2,
  # regularization rate
  reg = 1e-3,
  # show results every 'display' step
  display = 100,
  random.seed = 1)
{
  # to make the case reproducible.
  set.seed(random.seed)
  
  # total number of training set
  N <- nrow(traindata)
  
  # X = input layer, 1 row per observation, 1 col per input neuron
  # eg head(iris)
  #   Sepal.Length Sepal.Width Petal.Length Petal.Width
  # 1          5.1         3.5          1.4         0.2
  # 2          4.9         3.0          1.4         0.2
  # 3          4.7         3.2          1.3         0.2
  # 4          4.6         3.1          1.5         0.2 
  # 5          5.0         3.6          1.4         0.2 
  # 6          5.4         3.9          1.7         0.4
  
  # convert dataframe into datamatrix (only numerics, unnamed)
  X <- unname(data.matrix(traindata[,x]))
 
  # Y = expected outcomes as vector
  # eg head(iris$Species)
  # [1] setosa setosa setosa setosa setosa setosa
  # Levels: setosa versicolor virginica
  # eg as.integer(head(iris$Species))
  # [1] 1 1 1 1 1 1
  
    # correct categories represented by integer 
  Y <- traindata[,y]
  if(is.factor(Y)) { Y <- as.integer(Y) }
  Y.len   <- length(unique(Y))
  Y.set   <- sort(unique(Y))
  Y.index <- cbind(1:N, match(Y, Y.set))
  
  # create model or get model from parameter
  if(is.null(model)) {
    # number of input features
    D <- ncol(X)
    # number of categories for classification
    K <- length(unique(Y))
    H <-  hidden
    
    # create and init weights and bias 
    W1 <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
    b1 <- matrix(0, nrow=1, ncol=H)
    
    W2 <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
    b2 <- matrix(0, nrow=1, ncol=K)
  } else {
    D  <- model$D
    K  <- model$K
    H  <- model$H
    W1 <- model$W1
    b1 <- model$b1
    W2 <- model$W2
    b2 <- model$b2
  }
  
  report = data.frame(i=c(), loss=c(), accuracy=c())
  
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # init loss to a very big value
  loss <- 100000
  
  # Training the network
  i <- 0
  while(i < maxit && loss > abstol ) {
    
    # iteration index
    i <- i +1
    
    # Feed Forward
    # eg 3 input neurons to 5 hidden neurons
    # I1 I2 I3   Wi1h1 Wi1h2 Wi1h3 Wi1h4 Wi1h5    H1 H2 H3 H4 H5
    #          . Wi2h1 Wi2h2 Wi2h3 Wi2h4 Wi2h5  =
    #            Wi3h1 Wi3h2 Wi3h3 Wi3h4 Wi3h5
    
    # vectorized ((X dot W) + B)
    # 1 row per training observation
    hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    
    # Activation function : ReLU = max(neuron, 0)
    hidden.layer <- pmax(hidden.layer, 0) # pmax does max per element
    
    # output layer
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    
    # output layer activation function is softmax
    # softmax squashes range int 0..1 adding to 1
    # softmax = exp(output neuron i) / sum (exp(each output neuron))
    score.exp <- exp(score)
    probs <- score.exp/rowSums(score.exp)
    
    # compute the loss
    
    corect.logprobs <- -log(probs[Y.index]) #probs[Y.index] is the observed outcome
    # now correct.logprobs is a 1D array of the probability of the correct class for each training row
    
    # L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \frac{1}{2} \lambda \sum_k\sum_l W_{k,l}^2 }_\text{regularization loss} \\\\
    data.loss  <- sum(corect.logprobs)/batchsize # = average cross-entropy loss
    reg.loss   <- 0.5*reg* (sum(W1*W1) + sum(W2*W2))
    loss <- data.loss + reg.loss
    
    # display results and update model
    if( i %% display == 0) {
      if(!is.null(testdata)) {
        model <- list( D = D,
          H = H,
          K = K,
          # weights and bias
          W1 = W1, 
          b1 = b1, 
          W2 = W2, 
          b2 = b2)
        labs <- predict.dnn(model, testdata[,-y])
        accuracy <- mean(as.integer(testdata[,y]) == Y.set[labs])
        cat(i, loss, accuracy, "\n")
        report = rbind(report, data.frame(i=i, loss = loss, accuracy=accuracy))
      } else {
        cat(i, loss, "\n")
        report = rbind(report, data.frame(i=i, loss = loss, accuracy = NA))
      }
    }
    
    # backward ....
    
    # Calculate the derivate of the error/loss function
    # dE/dWji = (xi-ti)*xj
    # xi = softmax outputs = propbs
    # ti = 1 for the observed output = Y.index
    # Xj = final layer neuron activations
    dscores <- probs
    dscores[Y.index] <- dscores[Y.index] -1 # -1 for just the correct softmax scores
    dscores <- dscores / batchsize
    
    dW2 <- t(hidden.layer) %*% dscores 
    db2 <- colSums(dscores)
    
    dhidden <- dscores %*% t(W2)
    dhidden[hidden.layer <= 0] <- 0
    
    dW1 <- t(X) %*% dhidden
    db1 <- colSums(dhidden) 
    
    # update ....
    dW2 <- dW2 + reg*W2
    dW1 <- dW1  + reg*W1
    
    W1 <- W1 - lr * dW1
    b1 <- b1 - lr * db1
    
    W2 <- W2 - lr * dW2
    b2 <- b2 - lr * db2
    
    
    
  }
  
  # final results
  # creat list to store learned parameters
  # you can add more parameters for debug and visualization
  # such as residuals, fitted.values ...
  model <- list( D = D,
    H = H,
    K = K,
    # weights and bias
    W1= W1, 
    b1= b1, 
    W2= W2, 
    b2= b2,
    report=report)
  
  return(model)
}

########################################################################
# testing
#######################################################################
set.seed(1)

test.iris <- function() {
  # 1. split data into test/train
  samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
  
  # 2. train model
  ir.model <- train.dnn(x=1:4, y=5, traindata=iris[samp,], testdata=iris[-samp,], hidden=6, maxit=2000, display=50)
  
  # labels.dnn <- predict.dnn(ir.model, iris[-samp, -5])
  # table(iris[-samp,5], labels.dnn)
  # #          labels.dnn
  # #            1  2  3
  # #setosa     25  0  0
  # #versicolor  0 24  1
  # #virginica   0  0 25
  
  ir.model$report %>% 
    gather(measure, value, -i) %>% 
    ggplot(aes(x=i, y=value, color=measure)) +
      geom_point() +
      geom_line()
}

test.spiral <- function() {
  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = matrix(0, nrow=N*K, ncol=D)
  Y = matrix(0, nrow=N*K, ncol=1)
  for (j in 1:K) {
    ix = (N*(j-1)+1):((N*j))
    r = seq(0, 1, length=N)
    t = seq((j-1)*4, j*4, length=N) + (0.2*rnorm(N))
    X[ix,1] = r*sin(t)
    X[ix,2] = r*cos(t)
    Y[ix] = j
  }
  train = data.frame(X=X[,1],Y=X[,2],K=Y)
  
  ir.model <- train.dnn(x=1:2, y=3, traindata=train, testdata=train, hidden=100, maxit=10000, display=50, lr=1)
  
  # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  # ggplot(data.frame(X=X[,1], Y=X[,2], K=as.factor(Y)), aes(x=X, y=Y, color=K)) +
  #   geom_point()
  
  ir.model$report %>% 
    gather(measure, value, -i) %>% 
    ggplot(aes(x=i, y=value, color=measure)) +
    geom_point() +
    geom_line()
}


