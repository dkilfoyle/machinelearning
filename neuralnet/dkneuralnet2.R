# test4 = RPROP

library(stringr)

sigmoid = function(z) return(1.0/(1.0+exp(-z)))
sigmoid.prime = function(z) return(sigmoid(z)*(1-sigmoid(z)))

netlog = function(net, ...) {
  net$log = paste0(net$log, ...)
  return(net)
}

dksign = function(x) {
  if (abs(x) < 0.000001)
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
  maxepochs=500, maxerror=0.01, mini.batch.percent=10,
  epochUpdateFreq=10, randomEpoch=T,
  test.data=NULL) {
  
  n = length(training.data)
  if (mini.batch.percent==0)
    mini.batch.n = 1 # ie on-line training
  else
    mini.batch.n = round(mini.batch.percent/100 * n,0)
  net$MSE = numeric(maxepochs)
  
  net=netlog(net, "Training data: Length=",n,", Epochs=",maxepochs,", Batch Size=",mini.batch.n,"\n")
  
  # run throught the training data multiple times (epochs) in randomized order
  for (j in 1:maxepochs) {
    net$SSE = 0.0 # Squared Error for averaging
    
    # process training data in batches of mini.batch.size
    for (i in seq(1, n, mini.batch.n)) {
      
      # sample(1:n) randomizes the order of the training data
      if (randomEpoch) # randomise orderering of training data each epoch
        net = SGD.mini.batch(net, training.data[sample(1:n)[i:(i+mini.batch.n-1)]])
      else
        net=net = SGD.mini.batch(net, training.data[i:(i+mini.batch.n-1)])
    }
    
    net$MSE[j] = net$SSE/n

    if ((j==1) | ((j %% epochUpdateFreq)==0)) {
      
      # evaluate the net at the end of this epoch using test data
      if (!(is.null(test.data)))
        accuracy = paste0("Accuracy = ", round(evaluateNet(net, test.data)/length(test.data)*100,0), "%")
      else
        accuracy = ""
      
      net = net %>% 
        netlog(str_pad(paste0(" Epoch ", j, " "), 13, "right")) %>%
        netlog(str_pad(paste0("MSE = ", round(net$MSE[j],4), " "), 15, "right")) %>% 
        netlog(accuracy, "\n")
    }
    
    if (!is.null(net$progressFn)) net$progressFn(j/maxepochs)
    
    if (net$SSE < maxerror)
      break
  }
  
  return(net)
}

netTrainStep = function(net, training.data, method="Online", test.data=NULL) {
  if (net$step==0) {
    net$MSE = numeric(500) # 500 max steps
    net$SSE = 0
    net$step = 1
    net = netlog(net, "Training stepwise ",method,": length=", length(training.data), "\n")
  }

  if (method=="Online") {
    net = SGD.mini.batch(net, training.data[((net$step-1) %% length(training.data))+1])
    MSE = net$SSE/net$step
    if (net$step %% length(training.data) == 0) {
      # a new epoch
      net$SSE = 0
    }
  }
  else {
    net = SGD.mini.batch(net, training.data)
    MSE = net$SSE/length(training.data)
    net$SSE = 0
  }
  
  net = net %>% 
    netlog(str_pad(paste0(" Step ", net$step, " "), 9, "right")) %>% 
    netlog(str_pad(paste0("MSE = ", round(MSE, 4)), 15, "right"),"\n")
  
  net$MSE[j] = MSE
  net$step = net$step + 1
  
  return(net)
}

# Process a mini batch = for each x in mini batch:
# 1) feedForward to generate current activations
# 2) backPropogate to generate error gradients for each weight (partial derivative of error with respect to weight )
# 3) average the error gradients across the mini batch
# 4) update weights based on error gradient and learning rate eta
SGD.mini.batch = function(net, mini.batch) {
  
  # Zero the gradient vectors using the same shape as the source 
  net$gradient_sum.b = lapply(net$biases, function(x) x*0)
  net$gradient_sum.w = lapply(net$weights, function(x) x*0)
  
  # calculate and sum the gradient vector for each x in mini.batch
  m = length(mini.batch)
  for (i in 1:m) {
    
    x = mini.batch[[i]]
    
    # Feed forward
    net = net %>%
      feedForward(x[[1]]) %>% 
      backPropogate(x[[1]], x[[2]]) 
    
    # running total for all x in this minibatch
    for (l in 2:net$num.layers) {
      net$gradient_sum.b[[l]] = net$gradient_sum.b[[l]] + net$gradient.b[[l]]
      net$gradient_sum.w[[l]] = net$gradient_sum.w[[l]] + net$gradient.w[[l]]
    }
    
  }

  if (net$SGD.method=="standard")
    net = updateWeightsStandard(net)
  else if (net$SGD.method=="resilient")
    net = updateWeightsRPROP(net)
  else
    stop("Invalid method")
    
  return (net)
}

# feed training data forward generating a per layer list of activations and z values
feedForward = function(net, x) {
  activation = as.matrix(x) # input values
  net$activations = list(activation) # list to store all the activations layer by layer
  net$z = list() # list to store all the zs layer by layer, where z = wa+b
  
  #     I1  I2         
  # H1  w   w   dot  I1   =   H1z
  # H2  w   w        I2       H2z
  
  for (l in 2:net$num.layers) { # because layer 1 has no weights
    net$z[[l]] = (net$weights[[l]] %*% activation) + net$biases[[l]] # biases gets converted to column vector
    activation = sigmoid(net$z[[l]])
    net$activations[[l]] = activation # activations is a column vector
  }
  
  return(net)
}

# calculate the error gradients (gradient = pE/pw_ij) for each weight
# 1) Start with the activations and zs generated by feedForward
# 2) calculate the error for each neuron in the final layer
# 3) calculate nodedelta from error, sigmoidprime and for interior nodes the weights
# 4) calculate the gradient (gradient) from the nodedeltas and activations
backPropogate = function(net, x, y) {
  net$gradient.b = list()
  net$gradient.w = list()
  L = net$num.layers
  
  # Step 1: Calculate the Error
  # costderivative = Error = actual - expected
  E = net$activations[[L]]-y
  net$SSE = net$SSE + sum(E*E)
  
  # Step 2: Calculate a delta for each node
  #   a) Calculate delta for the output nodes
  #        Output delta = -E * f`(z)
  #   b) Calculate delta for interior nodes using delta from the node after
  #        Interior delta = f`(z)*sum_k(w_ki*delta_k)
  # Output layer deltas first
  net$delta=list()
  net$delta[[L]] = -E * sigmoid.prime(net$z[[L]])
  
  # Step 3: Calculate the individual gradients (gradient)
  # gradient = partial derivate dE/dw = delta_k * output_i
  # gradient_h1_to_o1 = 
  
  # gradient vector for the output layer
  net$gradient.b[[L]]=net$delta[[L]]
  net$gradient.w[[L]]=net$delta[[L]] %*% t(net$activations[[L-1]]) # convert activations from column vector to row vector
  
  # now calculate gradient vectors for all prior layers except layer 1
  for (l in (L-1):2) { # stop at layer 2 as layer 1 has no weights
    
    # recalculate delta from prior delta (deltal+1)
    net$delta[[l]] = (t(net$weights[[l+1]]) %*% net$delta[[l+1]]) * sigmoid.prime(net$z[[l]])
    
    # calculate the gradient vector using delta
    net$gradient.w[[l]] = net$delta[[l]] %*% t(net$activations[[l-1]])
    net$gradient.b[[l]] = net$delta[[l]]
  }
  
  return(net)
}


# Update the weights using standard backpropgation ie a fixed eta
updateWeightsStandard = function(net) {
  
  for (l in 2:net$num.layers) {
    wtChange.w = (net$eta * net$gradient_sum.w[[l]]) + (net$momentum * net$lastWtChanges.w[[l]])
    wtChange.b = (net$eta * net$gradient_sum.b[[l]]) + (net$momentum * net$lastWtChanges.b[[l]])
    
    net$weights[[l]] = net$weights[[l]] + wtChange.w
    net$biases[[l]] = net$biases[[l]] + wtChange.b
    
    net$lastWtChanges.w[[l]] = wtChange.w
    net$lastWtChanges.b[[l]] = wtChange.b
  }
  
  return(net)
}

# Update weights using Resilient Propogation (RPROP)
updateWeightsRPROP = function(net) {

  for (l in 2:net$num.layers) {
    net = getRPROPWtChanges.w(net, net$gradient_sum.w[[l]], l)
    net = getRPROPWtChanges.b(net, net$gradient_sum.b[[l]], l)
    
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
evaluateNet = function(net, test_data) {
  correct = 0
  for (i in 1:length(test_data)) {
    output = solveForward(net, test_data[[i]][[1]])
    correct = correct + (round(output) == test_data[[i]][[2]])
  }
  return(correct)
}

getEuclideanNorm = function(l) {
  
}
  
netInit <- function(sizes, sd.method="sqrtn") {
  # set.seed(12345)
  num.layers=length(sizes)
  sizes=sizes
  
  # generate a vector for each layer containing biases for each neuron. \
  # layer 1 doesnt need a bias
  if (sd.method=="nguyen.widrow") {
    biases = lapply(sizes, function(x) runif(x)-0.5) # range -0.5..0.5
  }
  else
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
    
    if (sd.method=="nguyen.widrow") {
      # init weights to range -0.5..0.5
      return(matrix(runif(sizes[2]*sizes[1])-0.5,
        nrow = sizes[l], ncol=sizes[l-1]))
    }
    else {
      return(matrix(rnorm(sizes[l]*sizes[l-1], sd=weight.sd),
        nrow = sizes[l],
        ncol = sizes[l-1])*0.01)
    }
  })
  
  if (sd.method=="nguyen.widrow") {
    
    # debug
    # H1 receiving weights
    weights[[2]][1,1] = 0.5172524211645029
    weights[[2]][1,2] =  -0.5258712726818855
    biases[[2]][1] =   0.8891383322123643

    # H2 receiving weights
    weights[[2]][2,1] =  -0.007687742622070948
    weights[[2]][2,2] =  -0.48985643968339754
    biases[[2]][2] =    -0.6610227585583137
    
    # first hidden layer only 
    beta = 0.7 * sizes[2]^(1/sizes[1])
    for (i in 1:sizes[2]) {
      # for each receiving neuron m_rs
      euclid.norm = sqrt(sum(weights[[2]][i,]*weights[[2]][i,])+(biases[[2]][i])^2)
      weights[[2]][i,] = beta*weights[[2]][i,]/euclid.norm
      biases[[2]][i] = beta*biases[[2]][i]/euclid.norm
    }
  }
  
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
  
  x=list(
    step = 0,
    log="",
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
  
  x=netlog(x, "Net initiated: Layers=", num.layers, ", Sizes=", paste(sizes,collapse=","),"\n")
  
  return(x %>% 
      netRPROPGradientDescent() %>% 
      netProgressFn())
}

testnet = function() {
  training=list()
  training[[1]]=list(c(0,0),c(0))
  training[[2]]=list(c(1,0),c(1))
  training[[3]]=list(c(0,1),c(1))
  training[[4]]=list(c(1,1),c(0))
  
  net=netInit(c(2,2,1), sd.method="nguyen.widrow")
  
  # # H1 receiving weights
  # net$weights[[2]][1,1] = -0.06782947598673161
  # net$weights[[2]][1,2] =  0.22341077197888182
  # net$biases[[2]][1] =   -0.4635107399577998
  # 
  # # H2 receiving weights
  # net$weights[[2]][2,1] =  0.9487814395569221
  # net$weights[[2]][2,2] =  0.46158711646254
  # net$biases[[2]][2] =    0.09750161997450091
  # 
  # # o1 receiving weights
  # net$weights[[3]][1,1] = -0.22791948943117624
  # net$weights[[3]][1,2] =  0.581714099641357
  # net$biases[[3]][1] =    0.7792991203673414
  
  # # H1 receiving weights
  # net$weights[[2]][1,1] = 0.5172524
  # net$weights[[2]][1,2] =  -0.5258712
  # net$biases[[2]][1] =   0.88913833
  # 
  # # H2 receiving weights
  # net$weights[[2]][2,1] =  -0.00768774
  # net$weights[[2]][2,2] =  -0.4898564
  # net$biases[[2]][2] =    -0.661022
  # 
  # # o1 receiving weights
  # net$weights[[3]][1,1] = 0.23773012320107711
  # net$weights[[3]][1,2] = 0.2200753094813884
  # net$biases[[3]][1] =    0.7792991203673414
  
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
  
   # net=netTrain(net, training, epochs=100, mini.batch.n=1, epochUpdateFreq=1, randomEpoch = F)
  net=netTrainStep(net, training)
  return(net)
  
  # x=netInit(c(2,2,1)) %>% 
  #   netTrain(training, epochs=100, epochUpdateFreq=1)
}




