# test4 = RPROP

library(stringr)
library(ggplot2)

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
  
  while (som$iteration <= som$maxIterations) {
    som = som %>% 
      somTrainStep(trainingData)
    plotsom(som)
  }
  
  som=som %>%
    somLog("Finished")
  
  return(som)
}
  
somTrainStep = function(som, trainingData) {

  som$iteration= som$iteration+1
  x=trainingData[sample(1:nrow(trainingData), size=1),] # choose 1 training sample randomly from samples
  som$worst = 0
  
  som = som %>%
    somLearn(x) %>% 
    somCorrect() %>% 
    somLog("Iteration: ", som$iteration, ", BMU distance: ", som$bmu$distance,"\n")
  
  return(som)
}

somLearn = function(som, x) {
  
  bmu = findBMU(som, x)
  bmuRow = ceiling(bmu$index/som$gridWidth)
  bmuCol = bmu$index - ((bmuRow-1) * som$gridWidth)
  
  for(wtindex in 1:nrow(som$weights)) {
      locationRow = ceiling(wtindex/som$gridWidth)
      locationCol = wtindex - ((locationRow-1) * som$gridWidth)
      
      deltaRow = bmuRow - locationRow
      deltaCol = bmuCol - locationCol
      
      deltaRow = deltaRow * deltaRow
      deltaCol = deltaCol * deltaCol;
      
      v = (deltaRow/(2.0*som$rbfWidth*som$rbfWidth)) + (deltaCol/(2.0*som$rbfWidth*som$rbfWidth))
      neighbor = exp(-v)
      
      d = x - som$weights[wtindex,]
      som$corrections[wtindex, ] = (d * neighbor * som$learningRate)
    
  }
  
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
# TODO Vectorise
findBMU = function(som, x) {

  bestIndex = -1
  bestDistance = 0
  worstDistance = 0
  
  for (i in 1:nrow(som$weights)) {
    
    delta = x - som$weights[i,] # subtract each column value
    dist = sqrt(sum(delta * delta))
    
    if (bestIndex==-1 | dist<bestDistance) {
      bestIndex = i
      bestDistance = dist
    }
    
  }
  return(list(index=bestIndex, distance=bestDistance))
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
  
  return(som)
}

somInit <- function(inputSize, gridWidth, gridHeight) {
   # set.seed(12345)
  
  x=list(inputSize=inputSize,
      gridWidth=gridWidth,
      gridHeight=gridHeight,
      log="",
      echo=T
    )
  
  som=x %>% 
    somSetLearningParameters() %>% 
    somClear() %>% 
    somLog("SOM initiated: Input Size=", inputSize, ", Grid=", gridWidth,"*",gridHeight,"\n") %>% 
    somProgressFn()
  
  return(som)
}

somSetLearningParameters <- function(som, maxIterations=100, startRate=0.8, endRate=0.003, startWidth=30, endWidth=5) {
  som$maxIterations=maxIterations
  som$startRate = startRate
  som$endRate = endRate
  som$startWidth = startWidth
  som$endWidth = endWidth
  som$rateStep = (startRate-endRate)/maxIterations
  som$widthStep = (startWidth-endWidth)/maxIterations
  return(som)
}

plotsom = function(som) {
  z=as.data.frame((som$weights+1)/2.0)
  z$x = 1+(0:(som$gridWidth * som$gridHeight-1) %% som$gridWidth)
  z$y = rep(1:50,each=50)
  
  ggplot(data=z, aes(x=x, y=y, fill=rgb(V1,V2,V3))) +
    geom_tile() +
    scale_fill_identity() +
    xlab("") +
    ylab("") +
    xlim(1,som$gridWidth) +
    ylim(1,som$gridHeight)
}


testsom = function() {
  samples=matrix(runif(15*3, min=-1.0, max=1.0),
    nrow=15,
    ncol=3)
  
  som=somInit(inputSize=3, gridWidth=50, gridHeight=50) %>% 
    somTrain(samples)

  return(som)
}




