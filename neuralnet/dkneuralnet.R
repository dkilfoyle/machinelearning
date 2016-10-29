# test4 = RPROP
dkdebug=T

sigmoid = function(z) return(1.0/(1.0+exp(-z)))
sigmoid.prime = function(z) return(sigmoid(z)*(1-sigmoid(z)))

dksign = function(x) {
  if (abs(x) < 0.0000000000001)
    return(0)
  else if (x > 0)
    return(1)
  else
    return(-1)
}

solveForward = function(a) {
  # feed the activations from the previous layer into the neurons of the next layer
  for (l in 2:num.layers) {
    z = (weights[[l]] %*% a)+biases[[l]]
    a = sigmoid(z)
  }
  return(a)
}

# Stochastic gradient descent
# Run throught the trianing data epoch number of times
# For each run process training data in batches of mini.batch.size
# only update the weights once per mini.batch using the batches average gradient vector
SGD = function(training.data, epochs, mini.batch.size, eta, momentum, test.data=NULL, progressFn=NULL) {
  
  n = length(training.data)
  MSE = numeric(epochs)
  epochUpdateFreq=50
  
  # run throught the training data multiple times (epochs) in randomized order
  for (j in 1:epochs) {
    MSE[j] = 0.0 # Squared Error for averaging
    
    # process training data in batches of mini.batch.size
    for (i in seq(1, n, mini.batch.size)) {
      # sample(1:n) randomizes the order of the training data
      if (dkdebug) # don't randomise order
        MSE[j] = MSE[j] + do.mini.batch(training.data[i:(i+mini.batch.size-1)], eta, momentum)
      else
        MSE[j] = MSE[j] + do.mini.batch(training.data[sample(1:n)[i:(i+mini.batch.size-1)]], eta, momentum)
    }
    
    MSE[j] = MSE[j]/n

    if ((j==1) | ((j %% epochUpdateFreq)==0)) {
      cat("Epoch ", j, " ",sep="")
      # evaluate the net at the end of this epoch using test data
      if (!(is.null(test.data)))
        cat(evaluate(test.data), "/",length(test.data))
      cat(" MSE = ", MSE[j], "\n", sep="")
    }
    
    if (!(is.null(progressFn))) progressFn(j/epochs)
  }
  
  return(MSE)
}  

# Process a mini batch = for each x in mini batch:
# 1) feedForward to generate current activations
# 2) backPropogate to generate error gradients for each weight (partial derivative of error with respect to weight )
# 3) average the error gradients across the mini batch
# 4) update weights based on error gradient and learning rate eta
do.mini.batch = function(mini.batch, eta, momentum) {
  
  # Zero the gradient vectors using the same shape as the source 
  nabla_sum.b = lapply(biases, function(x) x*0)
  nabla_sum.w = lapply(weights, function(x) x*0)
  
  MSE = 0.0 # total squared error for minibatch
  
  # calculate and sum the gradient vector for each x in mini.batch
  m = length(mini.batch)
  for (i in 1:m) {
    
    x = mini.batch[[i]]
    
    # Feed forward
    feedForward(x[[1]])
    
    # Back propogation
    # calculate gradient vector based on a single training sample x
    nabla.x = backPropogate(x[[1]], x[[2]]) 
    
    # running total for all x in this minibatch
    for (l in 2:num.layers) {
      nabla_sum.b[[l]] = nabla_sum.b[[l]] + nabla.x$b[[l]]
      nabla_sum.w[[l]] = nabla_sum.w[[l]] + nabla.x$w[[l]]
    }
    
    MSE = MSE + sum(nabla.x$E)
  }
  
  # calculate the average gradient for the minibatch
  # nabla_avg.w = lapply(nabla_sum.w, function(x) x/m)
  # nabla_avg.b = lapply(nabla_sum.b, function(x) x/m)
  
  updateWeights(nabla_sum.w, nabla_sum.b, eta, momentum)
  # updateWeightsRPROP(nabla_sum.w, nabla_sum.b)
    
  return (MSE)
}

# Update the weights using standard backpropgation ie a fixed eta
updateWeights = function(nabla.w, nabla.b, eta, momentum) {
  
  for (l in 2:num.layers) {
    wtChange.w = (eta * nabla.w[[l]]) + (momentum * lastWtChanges.w[[l]])
    wtChange.b = (eta * nabla.b[[l]]) + (momentum * lastWtChanges.b[[l]])
    
    weights[[l]] <<- weights[[l]] + wtChange.w
    biases[[l]] <<- biases[[l]] + wtChange.b
    
    lastWtChanges.w[[l]] <<- wtChange.w
    lastWtChanges.b[[l]] <<- wtChange.b
  }
}

# Update weights using Resilient Propogation (RPROP)
updateWeightsRPROP = function(nabla.w, nabla.b) {

  for (l in 2:num.layers) {
    wtChange.w = getRPROPWtChanges.w(nabla.w[[l]], l)
    wtChange.b = getRPROPWtChanges.b(nabla.b[[l]], l)
    
    weights[[l]] <<- weights[[l]] + wtChange.w
    biases[[l]] <<- biases[[l]] + wtChange.b
    
    lastWtChanges.w[[l]] <<- wtChange.w
    lastWtChanges.b[[l]] <<- wtChange.b
    
  }
}

getRPROPWtChanges.w = function(gradients, l) {
  wtChanges = matrix(nrow=nrow(gradients),ncol=ncol(gradients))
  
  for (i in 1:nrow(gradients)) {
    for (j in 1:ncol(gradients)) {
  
      gradient = gradients[i,j]
      lastGradient = lastGradients.w[[l]][i,j]
      gradientChange = dksign(gradient*lastGradient)
      
      if (gradientChange > 0) {
        # gradient has retained its sign
        # therefore, increase the eta to converge faster
        delta = min(updateValues.w[[l]][i,j] * 1.2, 50)
        weightChange = dksign(gradient) * delta # keep going in direction of gradient
        
        updateValues.w[[l]][i,j] <<- delta
        lastGradients.w[[l]][i,j] <<- gradient
      
      } else if (gradientChange < 0) {
        # gradient has changed its sign
        # we overshot the mark, the delta was too big
        delta = max(updateValues.w[[l]][i,j] * 0.5, 0.0000001) # reduce the delta
        weightChange = -lastWtChanges.w[[l]][i,j] # go back to previous weight before overshooting mark

        updateValues.w[[l]][i,j] <<- delta
        lastGradients.w[[l]][i,j] <<- 0 # force a gradientChange==0
      
      } else { 
        # change.w == 0
        # gradient hasn't changed at all
        delta = updateValues.w[[l]][i,j]
        weightChange = dksign(gradient) * delta
        
        lastGradients.w[[l]][i,j] <<- gradient
      }
      
      wtChanges[i,j] = weightChange
    }
  }
  return (wtChanges)
}

getRPROPWtChanges.b = function(gradients, l) {
  wtChanges = numeric(nrow(gradients))
  for (i in 1:nrow(gradients)) {
    
      gradient = gradients[i]
      lastGradient = lastGradients.b[[l]][i]
      gradientChange = dksign(gradient*lastGradient)
      
      if (gradientChange > 0) {
        # gradient has retained its sign
        # therefore, increase the eta to converge faster
        delta = min(updateValues.b[[l]][i] * 1.2, 50)
        weightChange = dksign(gradient) * delta # keep going in direction of gradient

        updateValues.b[[l]][i] <<- delta
        lastGradients.b[[l]][i] <<- gradient
        
      } else if (gradientChange < 0) {
        # gradient has changed its sign
        # we overshot the mark, the delta was too big
        delta = max(updateValues.b[[l]][i] * 0.5, 0.0000001) # reduce the delta
        weightChange = -lastWtChanges.b[[l]][i] # go back to previous weight before overshooting mark
        
        updateValues.b[[l]][i] <<- delta
        lastGradients.b[[l]][i] <<- 0 # force a gradientChange==0
        
      } else { 
        # change.b == 0
        # gradient hasn't changed at all
        delta = updateValues.b[[l]][i]
        weightChange = dksign(gradient) * delta
        
        lastGradients.b[[l]][i] <<- gradient
      }
      
      wtChanges[i]=weightChange
    }
  return (wtChanges)
}



# feed training data forward generating a per layer list of activations and z values
feedForward = function(x) {
  activation = x # input values
  activations <<- list(x) # list to store all the activations layer by layer
  z<<- list() # list to store all the zs layer by layer, where z = wa+b

  for (l in 2:num.layers) { # because layer 1 has no weights
    z[[l]] <<- (weights[[l]] %*% activation) + biases[[l]]
    activation = sigmoid(z[[l]])
    activations[[l]] <<- activation
  }
}

# calculate the error gradients (nabla = pE/pw_ij) for each weight
# 1) Start with the activations and zs generated by feedForward
# 2) calculate the error for each neuron in the final layer
# 3) calculate nodedelta from error, sigmoidprime and for interior nodes the weights
# 4) calculate the gradient (nabla) from the nodedeltas and activations
backPropogate = function(x, y) {
  
  # Zero the gradient vectors using the same shape as the source 
  nabla.b = lapply(biases, function(x) x*0)
  nabla.w = lapply(weights, function(x) x*0)
  
  # delta is the error term of the gradient vector
  delta=list()
  
  L = num.layers
  
  # output delta = costderivative * sigmoid derivative(z)
  # costderivative = Error = actual - expected
  E = activations[[L]]-y
  delta[[L]] = -E * sigmoid.prime(z[[L]])
  
  # gradient vector for the output layer
  nabla.b[[L]]=delta[[L]]
  nabla.w[[L]]=delta[[L]] %*% t(activations[[L-1]])
  
  # now calculate gradient vectors for all prior layers except layer 1
  for (l in (L-1):2) { # stop at layer 2 as layer 1 has no weights
    
    # recalculate deltal from prior delta (deltal+1)
    delta[[l]] = (t(weights[[l+1]]) %*% delta[[l+1]]) * sigmoid.prime(z[[l]])
    
    # calculate the gradient vector using delta
    nabla.w[[l]] = delta[[l]] %*% t(activations[[l-1]])
    nabla.b[[l]] = delta[[l]]
  }
  
  return(list(w=nabla.w, b=nabla.b, E=(E*E)))
}

# Return the number of test inputs for which the neural network outputs the correct result
evaluate = function(test_data) {
  correct = 0
  for (i in 1:length(test_data)) {
    output = solveForward(test_data[[i]][[1]])
    correct = correct + (round(output) == test_data[[i]][[2]])
  }
  return(correct)
}
  
initNetwork <- function(sizes) {
  set.seed(12345)
  num.layers<<-length(sizes)
  sizes<<-sizes
  
  # generate a vector for each layer containing biases for each neuron. \
  # layer 1 doesnt need a bias
  biases <<- lapply(sizes, rnorm)
  biases[[1]] <<- NA
  
  lastGradients.b <<- lapply(sizes, function(x) { rep(0, x) })
  lastGradients.b[[1]] <<- NA
  
  lastWtChanges.b <<- lapply(sizes, function(x) { rep(0, x) })
  lastWtChanges.b[[1]] <<- NA
  
  updateValues.b <<- lapply(sizes, function(x) { rep(0.1, x) })
  updateValues.b[[1]] <<- NA
  
  # generate a matrix w^l_{rs} for the weights connecting
  # sending   neuron s in layer l-1 to
  # receiving neuron j in layer l
  # to allow use of dot products the matrix is organized
  # 1 row for each receiving neuron r in layer l
  # 1 col for each sending neuron s in layer l-1

  weights <<- lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rnorm(sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  lastGradients.w <<- lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rep(0, sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  lastWtChanges.w <<- lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rep(0, sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  updateValues.w <<- lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rep(0.1, sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  activations <<- list()
  z <<- list()
  updateValues <<- list()
  lastDelta <<- list()
}

weights = list()



# # H1 receiving weights
# weights[[2]][1,1] = -0.06782947598673161
# weights[[2]][1,2] =  0.22341077197888182
#  biases[[2]][1] =   -0.4635107399577998
# 
# # H2 receiving weights
# weights[[2]][2,1] =  0.9487814395569221
# weights[[2]][2,2] =  0.46158711646254
#  biases[[2]][2] =    0.09750161997450091
# 
# # o1 receiving weights
# weights[[3]][1,1] = -0.22791948943117624
# weights[[3]][1,2] =  0.581714099641357
#  biases[[3]][1] =    0.7792991203673414

# iris.scale = scale(iris[,1:4])
# species.n = as.integer(iris$Species)
# training=list()
# for (i in 1:nrow(iris.scale)) {
#   y=c(0,0,0)
#   y[species.n[i]]=1
#   training[[i]] = list(
#     as.vector(iris.scale[i,1:4]),
#     y
#   )
# }




