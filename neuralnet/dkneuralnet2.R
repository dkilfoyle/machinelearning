# test4 = RPROP

sigmoid = function(z) return(1.0/(1.0+exp(-z)))
sigmoid.prime = function(z) return(sigmoid(z)*(1-sigmoid(z)))

mergeLists <- function (base_list, overlay_list, recursive = TRUE) {
  if (length(base_list) == 0)
    overlay_list
  else if (length(overlay_list) == 0)
    base_list
  else {
    merged_list <- base_list
    for (name in names(overlay_list)) {
      base <- base_list[[name]]
      overlay <- overlay_list[[name]]
      if (is.list(base) && is.list(overlay) && recursive)
        merged_list[[name]] <- mergeLists(base, overlay)
      else {
        merged_list[[name]] <- NULL
        merged_list <- append(merged_list,
          overlay_list[which(names(overlay_list) %in% name)])
      }
    }
    merged_list
  }
}

dksign = function(x) {
  if (abs(x) < 0.00000000000000001)
    return(0)
  else if (x > 0)
    return(1)
  else
    return(-1)
}

solveForward = function(net, a) {
  # feed the activations from the previous layer into the neurons of the next layer
  for (l in 2:net$num.layers) {
    z = (net$weights[[l]] %*% a)+net$biases[[l]]
    a = sigmoid(z)
  }
  return(a)
}

netStandardGradientDescent = function(net, eta=0.7, momentum=0.3) {
  net$eta=eta
  net$momentum=momentum
  net$SGD.method = "standard"
  return(net)
}

netRPROPGradientDescent = function(net) {
  net$SGD.method = "resilient"
  return(net)
}

netProgressFn = function(net, fn=NULL) {
  net$progressFn = fn
  return(net)
}

# Stochastic gradient descent
# Run throught the trianing data epoch number of times
# For each run process training data in batches of mini.batch.size
# only update the weights once per mini.batch using the batches average gradient vector
netTrain = function(net, training.data, 
  epochs=500, mini.batch.size=100,
  epochUpdateFreq=10, randomEpoch=T,
  test.data=NULL) {
  
  n = length(training.data)
  mini.batch.n = round(mini.batch.size/100 * n,0)
  MSE = numeric(epochs)
  
  cat("Training data: Length",n,", Epochs:",epochs,"Batch Size:",mini.batch.size,"\n")
  
  # run throught the training data multiple times (epochs) in randomized order
  for (j in 1:epochs) {
    MSE[j] = 0.0 # Squared Error for averaging
    
    # process training data in batches of mini.batch.size
    for (i in seq(1, n, mini.batch.n)) {
      # sample(1:n) randomizes the order of the training data
      if (randomEpoch) # randomise orderering of training data each epoch
        net = SGD.mini.batch(net, training.data[sample(1:n)[i:(i+mini.batch.n-1)]])
      else
        net = SGD.mini.batch(net, training.data[i:(i+mini.batch.n-1)])
      MSE[j] = MSE[j] + net$mbMSE
    }
    
    # MSE[j] = MSE[j]/n

    if ((j==1) | ((j %% epochUpdateFreq)==0)) {
      cat("Epoch ", j, " ",sep="")
      # evaluate the net at the end of this epoch using test data
      if (!(is.null(test.data)))
        cat(evaluate(test.data), "/",length(test.data))
      cat(" MSE = ", MSE[j], "\n", sep="")
    }
    
    if (!is.null(net$progressFn)) net$progressFn(j/epochs)
  }
  
  net$MSE = MSE
  
  return(net)
}  

# Process a mini batch = for each x in mini batch:
# 1) feedForward to generate current activations
# 2) backPropogate to generate error gradients for each weight (partial derivative of error with respect to weight )
# 3) average the error gradients across the mini batch
# 4) update weights based on error gradient and learning rate eta
SGD.mini.batch = function(net, mini.batch) {
  
  # Zero the gradient vectors using the same shape as the source 
  nabla_sum.b = lapply(net$biases, function(x) x*0)
  nabla_sum.w = lapply(net$weights, function(x) x*0)
  
  MSE = 0.0 # total squared error for minibatch
  
  # calculate and sum the gradient vector for each x in mini.batch
  m = length(mini.batch)
  for (i in 1:m) {
    
    x = mini.batch[[i]]
    
    # Feed forward
    net = feedForward(net, x[[1]])
    
    # Back propogation
    # calculate gradient vector based on a single training sample x
    nabla.x = backPropogate(net, x[[1]], x[[2]]) 
    
    # running total for all x in this minibatch
    for (l in 2:net$num.layers) {
      nabla_sum.b[[l]] = nabla_sum.b[[l]] + nabla.x$b[[l]]
      nabla_sum.w[[l]] = nabla_sum.w[[l]] + nabla.x$w[[l]]
    }
    
    MSE = MSE + sum(nabla.x$E)
  }
  
  net$mbMSE = MSE
  
  # calculate the average gradient for the minibatch
  # nabla_avg.w = lapply(nabla_sum.w, function(x) x/m)
  # nabla_avg.b = lapply(nabla_sum.b, function(x) x/m)
  
  net$nabla_sum.w = nabla_sum.w
  net$nabla_sum.b = nabla_sum.b
  
  if (net$SGD.method=="standard")
    net = updateWeightsStandard(net, nabla_sum.w, nabla_sum.b)
  else if (net$SGD.method=="resilient")
    net = updateWeightsRPROP(net, nabla_sum.w, nabla_sum.b)
  else
    stop("Invalid method")
    
  return (net)
}

# feed training data forward generating a per layer list of activations and z values
feedForward = function(net, x) {
  activation = x # input values
  net$activations = list(x) # list to store all the activations layer by layer
  net$z = list() # list to store all the zs layer by layer, where z = wa+b
  
  for (l in 2:net$num.layers) { # because layer 1 has no weights
    net$z[[l]] = (net$weights[[l]] %*% activation) + net$biases[[l]]
    activation = sigmoid(net$z[[l]])
    net$activations[[l]] = activation
  }
  
  return(net)
}

# calculate the error gradients (nabla = pE/pw_ij) for each weight
# 1) Start with the activations and zs generated by feedForward
# 2) calculate the error for each neuron in the final layer
# 3) calculate nodedelta from error, sigmoidprime and for interior nodes the weights
# 4) calculate the gradient (nabla) from the nodedeltas and activations
backPropogate = function(net, x, y) {
  
  # Zero the gradient vectors using the same shape as the source 
  nabla.b = lapply(net$biases, function(x) x*0)
  nabla.w = lapply(net$weights, function(x) x*0)
  
  # delta is the error term of the gradient vector
  delta=list()
  
  L = net$num.layers
  
  # output delta = costderivative * sigmoid derivative(z)
  # costderivative = Error = actual - expected
  E = net$activations[[L]]-y
  delta[[L]] = -E * sigmoid.prime(net$z[[L]])
  
  # gradient vector for the output layer
  nabla.b[[L]]=delta[[L]]
  nabla.w[[L]]=delta[[L]] %*% t(net$activations[[L-1]])
  
  # now calculate gradient vectors for all prior layers except layer 1
  for (l in (L-1):2) { # stop at layer 2 as layer 1 has no weights
    
    # recalculate deltal from prior delta (deltal+1)
    delta[[l]] = (t(net$weights[[l+1]]) %*% delta[[l+1]]) * sigmoid.prime(net$z[[l]])
    
    # calculate the gradient vector using delta
    nabla.w[[l]] = delta[[l]] %*% t(net$activations[[l-1]])
    nabla.b[[l]] = delta[[l]]
  }
  
  return(list(w=nabla.w, b=nabla.b, E=(E*E)))
}


# Update the weights using standard backpropgation ie a fixed eta
updateWeightsStandard = function(net, nabla.w, nabla.b) {
  
  for (l in 2:net$num.layers) {
    wtChange.w = (net$eta * nabla.w[[l]]) + (net$momentum * net$lastWtChanges.w[[l]])
    wtChange.b = (net$eta * nabla.b[[l]]) + (net$momentum * net$lastWtChanges.b[[l]])
    
    net$weights[[l]] = net$weights[[l]] + wtChange.w
    net$biases[[l]] = net$biases[[l]] + wtChange.b
    
    net$lastWtChanges.w[[l]] = wtChange.w
    net$lastWtChanges.b[[l]] = wtChange.b
  }
  
  return(net)
}

# Update weights using Resilient Propogation (RPROP)
updateWeightsRPROP = function(net, nabla.w, nabla.b) {

  for (l in 2:net$num.layers) {
    net = getRPROPWtChanges.w(net, nabla.w[[l]], l)
    net = getRPROPWtChanges.b(net, nabla.b[[l]], l)
    
    net$weights[[l]] = net$weights[[l]] + net$wtChange.w
    net$biases[[l]] = net$biases[[l]] + net$wtChange.b
    
    net$lastWtChanges.w[[l]] = net$wtChange.w
    net$lastWtChanges.b[[l]] = net$wtChange.b
    
  }
  
  return(net)
}

getRPROPWtChanges.w = function(net, gradients, l) {
  # store this layers wtchanges
  net$wtChange.w = matrix(nrow=nrow(gradients),ncol=ncol(gradients))
  
  for (i in 1:nrow(gradients)) {
    for (j in 1:ncol(gradients)) {
  
      gradient = gradients[i,j]
      lastGradient = net$lastGradients.w[[l]][i,j]
      gradientChange = dksign(gradient*lastGradient)
      
      if (gradientChange > 0) {
        # gradient has retained its sign
        # therefore, increase the eta to converge faster
        delta = min(net$updateValues.w[[l]][i,j] * 1.2, 50)
        weightChange = dksign(gradient) * delta # keep going in direction of gradient
        
        net$updateValues.w[[l]][i,j] = delta
        net$lastGradients.w[[l]][i,j] = gradient
      
      } else if (gradientChange < 0) {
        # gradient has changed its sign
        # we overshot the mark, the delta was too big
        delta = max(net$updateValues.w[[l]][i,j] * 0.5, 0.000001) # reduce the delta
        weightChange = -net$lastWtChanges.w[[l]][i,j] # go back to previous weight before overshooting mark

        net$updateValues.w[[l]][i,j] = delta
        net$lastGradients.w[[l]][i,j] = 0 # force a gradientChange==0
      
      } else { 
        # change.w == 0
        # no change to delta
        delta = net$updateValues.w[[l]][i,j]
        weightChange = dksign(gradient) * delta
        
        net$lastGradients.w[[l]][i,j] = gradient
      }
      
      net$wtChange.w[i,j] = weightChange
      
      # if (abs(weightChange) < 0.001)
      #   browser()
    }
  }
  return (net)
}

getRPROPWtChanges.b = function(net, gradients, l) {
  net$wtChange.b = numeric(nrow(gradients))
  
  for (i in 1:nrow(gradients)) {
    
      gradient = gradients[i]
      lastGradient = net$lastGradients.b[[l]][i]
      gradientChange = dksign(gradient*lastGradient)
      
      if (gradientChange > 0) {
        # gradient has retained its sign
        # therefore, increase the eta to converge faster
        delta = min(net$updateValues.b[[l]][i] * 1.2, 50)
        weightChange = dksign(gradient) * delta # keep going in direction of gradient

        net$updateValues.b[[l]][i] = delta
        net$lastGradients.b[[l]][i] = gradient
        
      } else if (gradientChange < 0) {
        # gradient has changed its sign
        # we overshot the mark, the delta was too big
        delta = max(net$updateValues.b[[l]][i] * 0.5, 0.0000001) # reduce the delta
        weightChange = -net$lastWtChanges.b[[l]][i] # go back to previous weight before overshooting mark
        
        net$updateValues.b[[l]][i] = delta
        net$lastGradients.b[[l]][i] = 0 # force a gradientChange==0
        
      } else { 
        # change.b == 0
        # gradient hasn't changed at all
        delta = net$updateValues.b[[l]][i]
        weightChange = dksign(gradient) * delta
        
        net$lastGradients.b[[l]][i] = gradient
      }
      
      net$wtChange.b[i]=weightChange
    }
  return (net)
}





# Return the number of test inputs for which the neural network outputs the correct result
evaluate = function(test_data) {
  correct = 0
  for (i in 1:length(test_data)) {
    output = solveForward(net, test_data[[i]][[1]])
    correct = correct + (round(output) == test_data[[i]][[2]])
  }
  return(correct)
}
  
netInit <- function(sizes, sd.method="sqrtn") {
  # set.seed(12345)
  num.layers=length(sizes)
  sizes=sizes
  
  # generate a vector for each layer containing biases for each neuron. \
  # layer 1 doesnt need a bias
  biases = lapply(sizes, rnorm)
  biases[[1]] = NA
  
  lastGradients.b = lapply(sizes, function(x) { rep(0, x) })
  lastGradients.b[[1]] = NA
  
  lastWtChanges.b = lapply(sizes, function(x) { rep(0, x) })
  lastWtChanges.b[[1]] = NA
  
  updateValues.b = lapply(sizes, function(x) { rep(0.1, x) })
  updateValues.b[[1]] = NA
  
  # generate a matrix w^l_{rs} for the weights connecting
  # sending   neuron s in layer l-1 to
  # receiving neuron j in layer l
  # to allow use of dot products the matrix is organized
  # 1 row for each receiving neuron r in layer l
  # 1 col for each sending neuron s in layer l-1
  
  weights = lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    if ((sd.method=="sqrtn") & (l!=num.layers))
      # squeeze the weights into a narrower range to prevent saturation of the output function
      weight.sd = 1/sqrt(sizes[l-1])
    else
      weight.sd = 1.0
    matrix(rnorm(sizes[l]*sizes[l-1], sd=weight.sd),
        nrow = sizes[l],
        ncol = sizes[l-1])
  })
  
  lastGradients.w = lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rep(0, sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  lastWtChanges.w = lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rep(0, sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  updateValues.w = lapply(1:num.layers, function(l) {
    if (l==1) return(matrix()) # layer 1 doesn't need weights
    matrix(rep(0.1, sizes[l]*sizes[l-1]),
      nrow = sizes[l],
      ncol = sizes[l-1]
    )
  })
  
  # activations = list()
  # z = list()
  # updateValues = list()
  # lastDelta = list()
  
  # # H1 receiving weights
  # weights[[2]][1,1] <<- -0.06782947598673161
  # weights[[2]][1,2] <<-  0.22341077197888182
  # biases[[2]][1] <<-   -0.4635107399577998
  # 
  # # H2 receiving weights
  # weights[[2]][2,1] <<-  0.9487814395569221
  # weights[[2]][2,2] <<-  0.46158711646254
  # biases[[2]][2] <<-    0.09750161997450091
  # 
  # # o1 receiving weights
  # weights[[3]][1,1] <<- -0.22791948943117624
  # weights[[3]][1,2]<<-  0.581714099641357
  # biases[[3]][1] <<-    0.7792991203673414
  
  x=list(
    sizes=sizes,
    num.layers=num.layers,
    
    weights=weights,
    biases=biases,
    
    lastGradients.w =lastGradients.w,
    lastGradients.b =lastGradients.b,
    
    lastWtChanges.w = lastWtChanges.w,
    lastWtChanges.b = lastWtChanges.b,
    
    updateValues.w =  updateValues.w,
    updateValues.b =  updateValues.b,
    
    activations = list(),
    z = list(),
    updateValues = list(),
    lastDelta = list()
  )
  
  return(x %>% 
      netRPROPGradientDescent() %>% 
      netProgressFn())
}



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

testnet = function() {
  training=list()
  training[[1]]=list(c(0,0),c(0))
  training[[2]]=list(c(1,0),c(1))
  training[[3]]=list(c(0,1),c(1))
  training[[4]]=list(c(1,1),c(0))
  
  net=netInit(c(2,2,1))
  
  # H1 receiving weights
  weights[[2]][1,1] = -0.06782947598673161
  weights[[2]][1,2] =  0.22341077197888182
   biases[[2]][1] =   -0.4635107399577998

  # H2 receiving weights
  weights[[2]][2,1] =  0.9487814395569221
  weights[[2]][2,2] =  0.46158711646254
   biases[[2]][2] =    0.09750161997450091

  # o1 receiving weights
  weights[[3]][1,1] = -0.22791948943117624
  weights[[3]][1,2] =  0.581714099641357
   biases[[3]][1] =    0.7792991203673414
   
   net=netTrain(net, training, epochs=100, epochUpdateFreq=1)
  
  # x=netInit(c(2,2,1)) %>% 
  #   netTrain(training, epochs=100, epochUpdateFreq=1)
}




