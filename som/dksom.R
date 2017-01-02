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
  som=somLog(som, "Samples: Length=", som$inputSize, "\n")
  dimnames(som$weights) = list(NULL, colnames(trainingData))
  
  while (som$iteration <= som$maxIterations) {
    som = som %>% 
      somTrainStep(trainingData)
  }
  
  som=som %>%
    somLog("Finished")
  
  return(som)
}
  
somTrainStep = function(som, trainingData) {
  x=trainingData[sample(1:nrow(trainingData), size=1),] # choose 1 training sample randomly from samples
  
  som = som %>%
    somLearn(x) %>% 
    somCorrect() %>% 
    somEvaluate(trainingData) %>% 
    somNext()

    return(som)
}

i2rc = function(index) {
  bmuRow = ceiling(index/50)
  bmuCol = index - ((bmuRow-1) * 50)
  cat(bmuRow, ", ", bmuCol,"\n")
}

somLearn = function(som, x) {
  
  bmu = findBMU(som, x)
  bmuRow = ceiling(bmu$index/som$gridWidth)
  bmuCol = bmu$index - ((bmuRow-1) * som$gridWidth)
  w22 = 2.0*som$rbfWidth*som$rbfWidth
  
  locationRow = ceiling(1:nrow(som$weights)/som$gridWidth)
  locationCol = 1:nrow(som$weights) - ((locationRow-1) * som$gridWidth)

  deltaRow = (locationRow-bmuRow)^2
  deltaCol = (locationCol-bmuCol)^2
  
  v = (deltaRow/w22) + (deltaCol/w22)
  neighbor = exp(-v)

  d = -1*sweep(som$weights,2,x) # same as d=x-weights
  som$corrections = (d * neighbor * som$learningRate)

  som$neighbor = neighbor
  som$bmu = bmu
  return(som)
}

somCorrect = function(som) {
  som$weights = som$weights + som$corrections
  
  som$learningRate = som$learningRate - som$rateStep
  som$rbfWidth = som$rbfWidth - som$widthStep
  return(som)
}

# return the index of the Best Matching Unit
# this is the neuron with the closest euclidean distance to the input numbers
findBMU = function(som, x) {
  delta = sweep(som$weights,2,x) # same as -1 * (x - weights) and the -1 gets squared out in next line
  distances = sqrt(rowSums(delta*delta))
  minDistIndex = which.min(distances)
  return(list(index=minDistIndex, distance=distances[minDistIndex]))
}

getBMUDistance = function(x, weights) {
  delta = sweep(weights,2,x) # same as -1 * (x - weights) and the -1 gets squared out in next line
  distances=sqrt(rowSums(delta*delta))
  return(distances[which.min(distances)])
}

getBMUIndex = function(x, weights) {
  delta = sweep(weights,2,x) # same as -1 * (x - weights) and the -1 gets squared out in next line
  distances=sqrt(rowSums(delta*delta))
  return(which.min(distances))
}

somEvaluate = function(som, trainingData) {
  if (som$iteration %% som$evaluateEveryN == 0) {
    bmus = apply(trainingData, 1, getBMUDistance, som$weights)
    som$meanBMUDistance = c(som$meanBMUDistance, mean(bmus))
    som = som %>% 
      somLog("Iteration: ", som$iteration, ", Mean Distance = ", round(mean(bmus),3),"\n")
  }
  return(som)
}

somNext = function(som) {
  som$iteration = som$iteration+1
  return(som)
}

somClear <- function(som) {
  
  # set weights to random numbers -1.0..1.0
  # each input will connect to every neuron in grid
  # the grid is flattened into a 1d array (rows)
  # the weight from each input is in each col
  
  som$weights = matrix(runif(som$gridWidth*som$gridHeight*som$inputSize, min=-1.0, max=1.0),
    nrow=som$gridWidth*som$gridHeight,
    ncol=som$inputSize)
  
  som$corrections = matrix(0, nrow=som$gridWidth*som$gridHeight, ncol=som$inputSize)
  
  som$learningRate = som$startRate
  som$rbfWidth = som$startWidth
  som$iteration = 0
  som$meanBMUDistance = c()
  
  return(som)
}

somInit <- function(inputSize, gridWidth, gridHeight, evaluateEveryN=10) {
   # set.seed(12345)
  
  x=list(inputSize=inputSize,
      gridWidth=gridWidth,
      gridHeight=gridHeight,
      log="",
      echo=T,
      evaluateEveryN=evaluateEveryN
    )
  
  som=x %>% 
    somSetLearningParameters() %>% 
    somClear() %>% 
    somLog("SOM initiated: Input Size=", inputSize, ", Grid=", gridWidth,"*",gridHeight,"\n") %>% 
    somProgressFn()
  
  return(som)
}

somSetLearningParameters <- function(som, maxIterations=1000, startRate=0.8, endRate=0.003, startWidth=30, endWidth=5) {
  som$maxIterations=maxIterations
  som$startRate = startRate
  som$endRate = endRate
  som$startWidth = startWidth
  som$endWidth = endWidth
  som$rateStep = (startRate-endRate)/maxIterations
  som$widthStep = (startWidth-endWidth)/maxIterations
  return(som)
}

plotNeighbor = function(som) {
  z=data.frame(z=som$neighbor)
  z$y = ceiling(1:length(som$neighbor)/som$gridWidth)
  z$x = 1:length(som$neighbor) - ((z$y-1) * som$gridWidth)
  
  bmuRow = ceiling(som$bmu$index/som$gridWidth)
  bmuCol = som$bmu$index - ((bmuRow-1) * som$gridWidth)
  
  ggplot(data=z, aes(x=x, y=y, fill=rgb(z,z,z))) +
    geom_tile() +
    scale_fill_identity() +
    xlab("") +
    ylab("") +
    xlim(0,som$gridWidth+1) +
    ylim(0,som$gridHeight+1) +
    coord_fixed() +
    geom_tile(aes(x=bmuCol, y=bmuRow), fill=rgb(1,0,0))

}

plotSOMFeature = function(som, feature) {
  z=as.data.frame((som$weights+1)/2.0)
  z$nodesRow = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$nodesCol = rep(1:som$gridWidth, each=som$gridHeight)
  
  if (feature=="RGB") {
    p = ggplot(data=z, aes(x=nodesCol, y=nodesRow, fill=rgb(R,G,B))) +
      scale_fill_identity()
  }
  else
    p = ggplot(data=z, aes_string(x="nodesCol", y="nodesRow", fill=feature))
  
  p +
    geom_tile(show.legend=F) +
    xlab("") +
    ylab("") +
    xlim(0,som$gridWidth+1) +
    ylim(0,som$gridHeight+1) +
    coord_fixed()
}

plotClusters = function(som, nClusters=4) {
  z = kmeans(som$weights, nClusters)
  z=data.frame(cluster=z$cluster)
  z$nodesRow = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$nodesCol = rep(1:som$gridWidth, each=som$gridHeight)
  
  z %>% 
    ggplot(aes(x=nodesCol, y=nodesRow)) +
    geom_tile(aes(fill=cluster), show.legend=F) +
    xlab("") +
    ylab("") +
    xlim(0,som$gridWidth+1) +
    ylim(0,som$gridHeight+1) +
    coord_fixed()
}

plotMeanBMU = function(som) {
  data.frame(iteration=1:length(som$meanBMUDistance)*10, distance=som$meanBMUDistance) %>% 
    ggplot(aes(x=iteration, y=distance)) +
    geom_line()
}

plotNodeCount = function(som, trainingData) {
  bmus = apply(trainingData, 1, getBMUIndex, som$weights)
  counts = numeric(som$gridWidth*som$gridHeight)
  for (i in 1:length(bmus)) {
    counts[bmus[i]] = counts[bmus[i]]+1
  }
  
  z=data.frame(counts=rescale(counts))
  z$nodesRow = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$nodesCol = rep(1:som$gridWidth, each=som$gridHeight)
  
  z %>% 
    ggplot(aes(x=nodesCol, y=nodesRow)) +
      geom_tile(aes(fill=counts), show.legend=F) +
      scale_fill_gradient() +
      xlab("") +
      ylab("") +
      xlim(0,som$gridWidth+1) +
      ylim(0,som$gridHeight+1) +
      coord_fixed()
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

