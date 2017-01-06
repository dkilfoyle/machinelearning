# test4 = RPROP

library(stringr)
library(ggplot2)
library(scales)

sigmoid = function(z) return(1.0/(1.0+exp(-z)))
sigmoid.prime = function(z) return(sigmoid(z)*(1-sigmoid(z)))

somLog = function(som, ...) {
  x=paste0(...)
  som$log = paste0(som$log, x)
  if (som$echo==T)
    cat(x)
  return(som)
}

somProgressFn = function(som, fn=NULL) {
  som$progressFn = fn
  return(som)
}

somTrain = function(som, trainingData, runName="test") { 
  
  som$runName = runName
  som=somLog(som, "Training samples = ", nrow(trainingData), "\n")
  dimnames(som$weights) = list(NULL, colnames(trainingData))
  
  while (som$iteration <= som$maxIterations) {
    x=trainingData[sample(1:nrow(trainingData), size=1),] # choose 1 training sample randomly from samples
    som = som %>% 
      somLearn(x) %>% 
      somCorrect() %>% 
      somEvaluate(trainingData) %>% 
      somNext()
  }
  
  som=som %>%
    somLog("Finished")
  
  return(som)
}
  
somTrainStep = function(som, trainingData) {
  x=trainingData[sample(1:nrow(trainingData), size=1),] # choose 1 training sample randomly from samples
  dimnames(som$weights) = list(NULL, colnames(trainingData))
  som$stepping=T
  
  som = som %>%
    somLearn(x) %>% 
    somCorrect() %>% 
    somEvaluate(trainingData) %>% 
    somNext()

    return(som)
}

i2rc = function(som, index) {
  bmuRow = ceiling(index/som$gridWidth)
  bmuCol = index - ((bmuRow-1) * som$gridWidths)
  cat(bmuRow, ", ", bmuCol,"\n")
}

# Find the weight corrections after processing a single training sample x
somLearn = function(som, x) {
  
  bmu = findBMU(som, x)

  deltaRow = (som$indices$R-bmu$nodeRow)^2
  deltaCol = (som$indices$C-bmu$nodeCol)^2
  nodeDistSquared = deltaRow + deltaCol
  theta_t = exp(-nodeDistSquared/som$rbfWidthSqTimes2)
  
  weightDeltas = -1*sweep(som$weights,2,x) # same as d=x-weights
  som$corrections = (weightDeltas * theta_t * som$learningRate)

  som$neighbor = theta_t
  som$bmu = bmu
  return(som)
}

somCorrect = function(som) {
  som$weights = som$weights + som$corrections
  
  # learning rate exponential decay
  som$learningRate = som$startRate * exp(-som$iteration/som$maxIterations)
  
  # radial basis function width exponential decay
  som$rbfWidth = som$startWidth * exp(-som$iteration / (som$maxIterations/log(som$startWidth)))
  som$rbfWidthSqTimes2 = 2.0*(som$rbfWidth^2)
  
  return(som)
}

# return the index of the Best Matching Unit
# this is the neuron with the closest euclidean distance to the input numbers
findBMU = function(som, x) {
  delta = sweep(som$weights,2,x) # same as -1 * (x - weights) and the -1 gets squared out in next line
  distances = rowSums(delta*delta)
  minDistIndex = which.min(distances)
  return(list(index=minDistIndex, 
              distanceSq=distances[minDistIndex],
              nodeRow = som$indices$R[minDistIndex],
              nodeCol = som$indices$C[minDistIndex]))
}

getBMUDistance = function(x, weights) {
  delta = sweep(weights,2,x) # same as -1 * (x - weights) and the -1 gets squared out in next line
  distances=rowSums(delta*delta)
  return(distances[which.min(distances)])
}

getBMUIndex = function(x, weights) {
  delta = sweep(weights,2,x) # same as -1 * (x - weights) and the -1 gets squared out in next line
  distances=rowSums(delta*delta)
  return(which.min(distances))
}

somEvaluate = function(som, trainingData) {
  if (som$iteration %% som$evaluateFrequency == 0) {
    bmus = apply(trainingData[sample(1:nrow(trainingData), size=som$evaluateSampleProp*nrow(trainingData)),], 1, getBMUDistance, som$weights)
    som$meanBMUDistance = c(som$meanBMUDistance, mean(bmus))
    som = som %>% 
      somLog("Iteration: ", som$iteration, ", Mean Distance Sq = ", round(mean(bmus),3),"\n")
  }
  return(som)
}

somNext = function(som) {
  som$iteration = som$iteration+1
  if (!is.null(som$progressFn)) som$progressFn(som$iteration/som$maxIterations)
  return(som)
}



somInit <- function(inputSize, gridWidth, gridHeight, evaluateFrequency=10, evaluateSampleProp=1.0) {
   # set.seed(12345)
  
  x=list(inputSize=inputSize,
      gridWidth=gridWidth,
      gridHeight=gridHeight,
      log="",
      echo=T,
      evaluateFrequency=evaluateFrequency,
      evaluateSampleProp=evaluateSampleProp,
      stepping=F
    )
  
  x$indices = data.frame(
    R = rep(1:gridHeight, each=gridWidth),
    C = rep(1:gridWidth, times=gridHeight)
  )
  
  som=x %>% 
    somSetLearningParameters() %>% 
    somClear() %>% 
    somLog("SOM initiated: Input Dimensions=", inputSize, ", Grid=", gridWidth,"*",gridHeight,"\n") %>% 
    somProgressFn()
  
  return(som)
}

somSetLearningParameters <- function(som, maxIterations=1000, startRate=0.8, startWidth=30) {
  som$maxIterations=maxIterations
  som$startRate = startRate
  som$startWidth = startWidth
  return(som)
}

somClear <- function(som) {
  
  # set weights to random numbers -1.0..1.0
  # each input will connect to every neuron in grid
  # the grid is flattened into a 1d array (rows)
  # the weight from each input is in each col
  
  som$weights = matrix(runif(som$gridWidth*som$gridHeight*som$inputSize, min=-2.0, max=2.0),
                       nrow=som$gridWidth*som$gridHeight,
                       ncol=som$inputSize)
  
  som$corrections = matrix(0, nrow=som$gridWidth*som$gridHeight, ncol=som$inputSize)
  
  som$learningRate = som$startRate
  som$rbfWidth = som$startWidth
  som$rbfWidthSqTimes2 = 2.0*(som$rbfWidth^2)
  som$iteration = 0
  som$meanBMUDistance = c()
  
  return(som)
}


plotNeighbor = function(som) {
  z=data.frame(z=som$neighbor)
  z$y = som$indices$R #ceiling(1:length(som$neighbor)/som$gridWidth)
  z$x = som$indices$C #1:length(som$neighbor) - ((z$y-1) * som$gridWidth)
  
  bmuRow = som$indices$R[som$bmu$index] #ceiling(som$bmu$index/som$gridWidth)
  bmuCol = som$indices$C[som$bmu$index] #som$bmu$index - ((bmuRow-1) * som$gridWidth)
  
  ggplot(data=z, aes(x=x, y=y, fill=rgb(z,z,z))) +
    geom_tile() +
    scale_fill_identity() +
    labs(x=NULL, y=NULL) +
    xlim(0,som$gridWidth+1) +
    ylim(0,som$gridHeight+1) +
    coord_fixed() +
    geom_tile(aes(x=bmuCol, y=bmuRow), fill=rgb(1,0,0)) +
    theme_minimal() +
    ggtitle("Neighbors")

}

plotSOMFeature = function(som, feature) {
  z=as.data.frame(som$weights)
  z$nodesRow = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$nodesCol = rep(1:som$gridWidth, each=som$gridHeight)
  
  if (feature=="RGB") {
    p = ggplot(data=z, aes(x=nodesCol, y=nodesRow, fill=rgb(rescale(R),rescale(G),rescale(B)))) +
      scale_fill_identity()
  }
  else {
    p = ggplot(data=z, aes_string(x="nodesCol", y="nodesRow", fill=feature)) +
    scale_fill_gradientn(colours=rev(rainbow(100,end=4/6)))
  }
  
  p +
    geom_tile(show.legend=F) +
    labs(x=NULL,y=NULL) +
    xlim(0,som$gridWidth+1) +
    ylim(0,som$gridHeight+1) +
    coord_fixed() +
    theme_minimal() +
    ggtitle(feature)
}

plotClusters = function(som, nClusters=4) {
  z = kmeans(som$weights, nClusters)
  z=data.frame(cluster=z$cluster)
  z$nodesRow = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$nodesCol = rep(1:som$gridWidth, each=som$gridHeight)
  
  z %>% 
    ggplot(aes(x=nodesCol, y=nodesRow)) +
    geom_tile(aes(fill=cluster), show.legend=F) +
    labs(x=NULL,y=NULL) +
    xlim(0,som$gridWidth+1) +
    ylim(0,som$gridHeight+1) +
    coord_fixed() +
    theme_minimal()
}

plotMeanBMU = function(som) {
  data.frame(iteration=1:length(som$meanBMUDistance)*10, distance=som$meanBMUDistance) %>% 
    ggplot(aes(x=iteration, y=distance)) +
    geom_line()
}

# count the number of samples that BMU to each node
plotBMUCount = function(som, trainingData) {
  bmus = apply(trainingData, 1, getBMUIndex, som$weights)
  counts = numeric(som$gridWidth*som$gridHeight)
  for (i in 1:length(bmus)) {
    counts[bmus[i]] = counts[bmus[i]]+1
  }
  
  z=data.frame(counts=counts)
  z$nodesRow = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$nodesCol = rep(1:som$gridWidth, each=som$gridHeight)
  
  z %>% 
    ggplot(aes(x=nodesCol, y=nodesRow)) +
      geom_tile(aes(fill=counts), show.legend=T) +
      scale_fill_gradientn(colours=rev(rainbow(4000,end=4/6))) +
      labs(x=NULL,y=NULL) +
      xlim(0,som$gridWidth+1) +
      ylim(0,som$gridHeight+1) +
      coord_fixed() +
      theme_minimal() +
      ggtitle("BMUs per Node")
}

testsom = function() {
  samples=matrix(runif(15*3, min=-1.0, max=1.0),
    nrow=15,
    ncol=3)
  dimnames(samples) = list(NULL, c("R","G","B"))
  
  som=somInit(inputSize=3, gridWidth=50, gridHeight=50) %>% 
    somTrain(samples)

  return(som)
}

testdublin = function() {
  mydata = as.matrix(readRDS("census.Rda")[,c(2,4,5,8)])
  # mydata= rescale(mydata, to=c(-1,1))
  mydata=scale(mydata)
  som=somInit(inputSize=4, gridWidth=50, gridHeight=50, evaluateFrequency=100, evaluateSampleProp=0.1) %>% 
    somSetLearningParameters(maxIterations=1, startRate = 0.05, endRate = 0.001, startWidth = 10, endWidth=2) %>% 
    somClear
    somTrain(mydata)
}

# mydata = as.matrix(scale(readRDS("census.Rda")[,c(2,4,5,8)]))
# x=somInit(inputSize=4, gridWidth=50, gridHeight=50, evaluateFrequency=100, evaluateSampleProp=0.1)

