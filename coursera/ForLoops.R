nnCostFunctionForLoop = function(input_layer_size, hidden_layer_size, num_labels,X, y, lambda=0) {
  function(nn_params) {
    # Reshape the unrolled thetas into the weight matrices
    Theta1_size = hidden_layer_size * (input_layer_size + 1)
    Theta1 = matrix(nn_params[1:Theta1_size], nrow=hidden_layer_size, ncol=(input_layer_size+1))
    Theta2 = matrix(nn_params[(Theta1_size+1):length(nn_params)], nrow=num_labels, ncol=(hidden_layer_size+1))
    
    # m is the number of samples (rows) in X
    m <- nrow(X)
    J = 0
    
    # using for loop to sum cost i..m by theta_l %*% a_l
    # O10 O11 O12 O13 O14   a0=1     z1
    # O20 O21 O22 O23 O24   a1       z2
    # O30 O31 O32 O33 O34   a2    =  z3
    #                       a3
    #                       a4
    for (i in 1:m) {
      y_k = diag(num_labels)[y[i],]   # eg y=3 = c(0,0,1,0,0,0,0,0,0,0)
      a1 = matrix(c(1, X[i, ]))
      z2 = Theta1 %*% a1         # input to hidden
      a2 = matrix(c(1, sigmoid(z2)))   # add a_0^(2)
      z3 = Theta2 %*% a2         # hidden to output
      a3 = sigmoid(z3)
      h  = a3
      # vectorise the sum over k labels
      J = J + 1/m * (t(-y_k) %*% log(h) - (t(1-y_k) %*% log(1-h)))
    }
    J = J + (lambda / (2 * m)) * (sum(Theta1[,-1]^2) + sum(Theta2[,-1]^2)) # regularisation cost
    
    return(J)
  }
}

nnGradientFunctionForLoop = function(input_layer_size, hidden_layer_size, num_labels, X, y, lambda=0) {
  function(nn_params) {
    # Reshape the unrolled thetas into the weight matrices
    Theta1_size = hidden_layer_size * (input_layer_size + 1)
    Theta1 = matrix(nn_params[1:Theta1_size], nrow=hidden_layer_size, ncol=(input_layer_size+1))
    Theta2 = matrix(nn_params[(Theta1_size+1):length(nn_params)], nrow=num_labels, ncol=(hidden_layer_size+1))
    
    # m is the number of samples (rows) in X
    m <- nrow(X)
    J = 0
    
    # You need to return the following variables correctly
    Theta1_grad = matrix(0, nrow=nrow(Theta1), ncol=ncol(Theta1))
    Theta2_grad = matrix(0, nrow=nrow(Theta2), ncol=ncol(Theta2))
    
    for (i in 1:m) {
      y_k = diag(num_labels)[y[i],]   # eg y=3 = c(0,0,1,0,0,0,0,0,0,0)
      a1 = matrix(c(1, X[i, ]))
      z2 = Theta1 %*% a1         # input to hidden
      a2 = matrix(c(1, sigmoid(z2)))   # add a_0^(2)
      z3 = Theta2 %*% a2         # hidden to output
      a3 = sigmoid(z3)
      h  = a3
      
      # compute delta for output layer
      delta_l3 = a3 - y_k
      
      # compute delta for the hidden layer and drop the bias term
      delta_l2 = ((t(Theta2) %*% delta_l3) * a2 * (1-a2))[-1]
      
      Theta2_grad = Theta2_grad + (delta_l3 %*% t(a2))
      Theta1_grad = Theta1_grad + (delta_l2 %*% t(a1))
    }
    
    # average the Delta and add regularisation
    Theta1_0bias = Theta1
    Theta1_0bias[,1] = rep(0, nrow(Theta1))
    Theta2_0bias = Theta2
    Theta2_0bias[,1] = rep(0, nrow(Theta2))
    D1 = 1/m * Theta1_grad + lambda * Theta1_0bias # exclude the bias column in theta
    D2 = 1/m * Theta2_grad + lambda * Theta2_0bias
    
    grad=c(D1, D2)
    return(grad)
  }
}