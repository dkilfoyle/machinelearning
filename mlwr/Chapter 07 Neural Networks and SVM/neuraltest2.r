sigmoid = function(z) return(1.0/(1.0+exp(-z)))
sigmoid.prime = function(z) return(sigmoid(z)*(1-sigmoid(z)))

feedForward = function(a) {
  # feed the activations from the previous layer into the neurons of the next layer
  for (l in 2:num.layers) {
    z = (weights[[l]] %*% a)+biases[[l]]
    a = sigmoid(z)
    ffz[[l]] <<- z
    ffa[[l]] <<- a
  }
  return(a)
}

# Stochastic gradient descent
# Do gradient descent on epochs of mini.batch.size instead of the entire test data
# This makes calculating the cost gradient much less intensive
SGD = function(training.data, epochs, mini.batch.size, eta, test.data=NULL) {
  
  n = length(training.data)
  
  # run throught the training data multiple times (epochs) in randomized order
  for (j in 1:epochs) {
    cat("Epoch ", j, " ")
    
    # split the training data into batches of mini.batch.size
    MSE = 0.0 # Squared Error for averaging
    for (i in seq(1, n, mini.batch.size)) {
      # sample(1:n) randomizes the order of the training data
      MSE = MSE + do.mini.batch(training.data[sample(1:n)[i:(i+mini.batch.size-1)]], eta)
    }

    if (!(is.null(test.data))) {
      n.test=length(test.data)
      cat(evaluate(test.data)," / ",n.test, " MSE = ", MSE/n, " Output = ", activations[[3]][1])
    }

    cat("\n")
  }
}  

# Update network weights and biases by applying gradient descent backpropogation to a single minibatch
do.mini.batch = function(mini.batch, eta) {
  
  # Zero the gradient vectors using the same shape as the source 
  nabla.b.sum = lapply(biases, function(x) x*0)
  nabla.w.sum = lapply(weights, function(x) x*0)
  
  MSE = 0.0 # total squared error for minibatch
  
  # calculate and sum the gradient vector for each x in mini.batch
  m = length(mini.batch)
  for (i in 1:m) {
    
    # backpropagation ==============================================================
    # calculate gradient vector based on a single training sample x
    nabla.x = backprop(mini.batch[[i]][[1]], mini.batch[[i]][[2]]) 
    
    # running total for all x in minibatch
    for (l in 2:num.layers) {
      nabla.b.sum[[l]] = nabla.b.sum[[l]] + nabla.x$b[[l]]
      nabla.w.sum[[l]] = nabla.w.sum[[l]] + nabla.x$w[[l]]
    }
    
    MSE = MSE + sum(nabla.x$E)
  }
  
  # Update the weights using the gradient vector and eta
  # Use the average of gradient vectors in the mini.batch
  # ie nabla.sum / m
  # see eq 10 and 11
  # delta_w = eta * (sum.of.nablas / m)
  # new_w = old_w + delta_w
  eta.div.m = eta/m
  for (l in 2:num.layers) {
    weights[[l]] <<- weights[[l]] + (eta.div.m * nabla.w.sum[[l]])
    biases[[l]] <<- biases[[l]] + (eta.div.m * nabla.b.sum[[l]])
  }
  
  return(MSE)
}

# return a list of nabla.b and nabla.w which are the pC/pw and pC/pb calculated from the activations and the errors
backprop = function(x, y) {
  
  # feedforward
  # ===========
  
  activation = x # input values
  activations <<- list(x) # list to store all the activations layer by layer
  z = list() # list to store all the zs layer by layer, where z = wa+b
  
  L = num.layers 
  
  for (l in 2:L) { # because layer 1 has no weights
    z[[l]] = (weights[[l]] %*% activation) + biases[[l]]
    activation = sigmoid(z[[l]])
    activations[[l]] <<- activation
  }
  
  # Back propagation
  # ================
  
  # Zero the gradient vectors using the same shape as the source 
  nabla.b = lapply(biases, function(x) x*0)
  nabla.w = lapply(weights, function(x) x*0)
  
  # delta is the error term of the gradient vector
  delta=list()
  
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

evaluate = function(test_data) {
  # Return the number of test inputs for which the neural
  # network outputs the correct result. Note that the neural
  # network's output is assumed to be the index of whichever
  # neuron in the final layer has the highest activation.
  correct = 0
  
  for (i in 1:length(test_data)) {
    output = feedForward(test_data[[i]][[1]])
    correct = correct + (round(output) == test_data[[i]][[2]])
  }
  
  return(correct)
}
  
set.seed(12345)
sizes=c(2,3,1)
num.layers=length(sizes)

# generate a vector for each layer containing biases for each neuron. \
# layer 1 doesnt need a bias
biases = lapply(sizes, rnorm)
biases[[1]]=NA

# generate a matrix w^l_{rs} for the weights connecting
# sending   neuron s in layer l-1 to
# receiving neuron j in layer l
# to allow use of dot products the matrix is organized
# 1 row for each receiving neuron r in layer l
# 1 col for each sending neuron s in layer l-1
weights = lapply(1:num.layers, function(l) {
  if (l==1) return(matrix()) # layer 1 doesn't need weights
  matrix(rnorm(sizes[l]*sizes[l-1]),
    nrow = sizes[l],
    ncol = sizes[l-1]
  )
})

# # H1 receiving weights
# weights[[2]][1,1] = -0.06782
# weights[[2]][1,2] =  0.22341
#  biases[[2]][1] =   -0.46351
# 
# # H2 receiving weights
# weights[[2]][2,1] =  0.94878
# weights[[2]][2,2] =  0.46158
#  biases[[2]][2] =    0.09750
# 
# # o1 receiving weights
# weights[[3]][1,1] = -0.227919
# weights[[3]][1,2] =  0.581714
#  biases[[3]][1] =    0.779299

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

training=list()
training[[1]]=list(c(0,0),c(0))
training[[2]]=list(c(1,0),c(1))
training[[3]]=list(c(0,1),c(1))
training[[4]]=list(c(1,1),c(0))

E = list()

SGD(training, 10000, 1, 0.7, training)

# ffz = list()
# ffa = list()
# feedForward(c(1.0,0.0))
