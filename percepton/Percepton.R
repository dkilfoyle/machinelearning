library(R6)
library(ggplot2)
library(dplyr)

# Code adapted from python source in Python Machine Learning by Sebastian Raschka

Percepton = R6Class("Percepton",
  public=list(
    w = NULL,
    b = NULL,
    eta = NULL,
    n_iter = NULL,
    ierrors = NULL,
    
    initialize = function(eta=0.01, n_iter=10) {
      self$eta = eta
      self$n_iter = n_iter
    },
    
    fit = function(X, y) {
      self$w = rep(0, ncol(X)) # initialize w to 0s
      self$b = 0
      self$ierrors = c()
      
      for (i in 1:self$n_iter) {
        errors = 0
        for (xi in 1:nrow(X)) {
          x = as.numeric(X[xi,])
          update = self$eta * (y[xi] - self$predict(x))
          self$w = self$w + (update * x)
          self$b = self$b + update
          errors = errors + as.integer(update != 0.0)
        }
        self$ierrors = append(self$ierrors, errors)
      }
      return(self)
    },
    
    net_input = function(x) {
      dotp = as.numeric(x %*% self$w)
      return(dotp + self$b)
    },
    
    predict = function(x) {
      return(ifelse(self$net_input(x)>=0.0,1,-1))
    }
  )
)

plot_decision_regions = function(X,y,classifier,resolution=0.02) {
  library(pracma)
  mg = meshgrid(
    seq(min(X[,1])-1, max(X[,1])+1, resolution),
    seq(min(X[,2])-1, max(X[,2])+1, resolution))
  
  ds = data.frame(x=as.vector(mg$X), y=as.vector(mg$Y))
  ds$z = apply(ds,1,function(x) classifier$predict(x))
  
  p = ggplot(ds, aes(x=x, y=y)) + 
    geom_tile(aes(fill=as.factor(z)), show.legend=F) +
    scale_fill_manual(values=alpha(c("Blue","Red"),0.2)) +
    geom_point(
      data=data.frame(x=X[,1], y=X[,2], z=as.factor(y)),
      aes(x=x, y=y, color=z, shape=z))
  
  return(p)
}

AdalineGD = R6Class("AdalineGD",
  public=list(
    w = NULL,
    b = NULL,
    eta = NULL,
    n_iter = NULL,
    cost = NULL,
    output = NULL,
    errors = NULL,
    
    initialize = function(eta=0.01, n_iter=10) {
      self$eta = eta
      self$n_iter = n_iter
    },
    
    fit = function(X, y) {
      self$w = rep(0, ncol(X)) # initialize w to 0s
      self$b = 0
      self$cost = c()
      
      for (i in 1:self$n_iter) {
        output = apply(X, 1, self$net_input)
        errors = (y - output)
        # weight change = -n * gradient of cost
        # gradient of sum squared error is -sum(y-output)xi
        self$w = self$w + self$eta * (t(X) %*% errors)
        self$b = self$b + self$eta * sum(errors)
        cost = sum((errors * errors)) / 2.0
        self$cost = append(self$cost, cost)
      }
      return(self)
    },
    
    net_input = function(x) {
      dotp = as.numeric(x %*% self$w)
      return(dotp + self$b)
    },
    
    predict = function(x) {
      return(ifelse(self$net_input(x)>=0.0,1,-1))
    }
  )
)

# test = iris %>% 
#   slice(1:100) %>% 
#   mutate(y = ifelse(Species=="setosa",-1,1))

# ppn2 = AdalineGD$new(eta=0.01, n_iter=15)
# ppn2$fit(scale(test[,c(1,3)]), test$y)



