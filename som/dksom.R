# test4 = RPROP

library(stringr)

sigmoid = function(z) return(1.0/(1.0+exp(-z)))
sigmoid.prime = function(z) return(sigmoid(z)*(1-sigmoid(z)))

somlog = function(som, ...) {
  som$log = paste0(som$log, ...)
  return(som)
}

somProgressFn = function(som, fn=NULL) {
  som$progressFn = fn
  return(som)
}

somTrain = function(som, samples, runName) { 
  
  som$runName = runName
  som=somlog(som, "Samples: Length=", som$inputSize, "\n")
  
  while (som$iteration <= som$maxIterations) {
    som = som %>% 
      somTrainStep(samples)
  }
  
  som=som %>%
    somlog("Finished")
  
  return(som)
}
  
somTrainStep = function(som, samples) {
  som$iteration= som$iteration+1
  
  sample1=sample(samples,size=1)
  
  som = som %>%
    somLearn(sample1) %>% 
    somCorrect()
}

# Return the number of test inputs for which the neural somwork outputs the correct result
evaluatesom = function(som, test_data) {
}

somInit <- function(inputSize, gridWidth, gridHeight) {
 
   # set.seed(12345)
  
  # set weights to random numbers -1.0..1.0
  weights = matrix(runif(gridWidth*gridHeight*inputSize, min=-1.0, max=1.0),
    nrow=gridWidth*gridHeight,
    ncol=inputSize)
  
  x=list(
      weights=weights,
      inputSize=inputSize,
      gridWidth=gridWidth,
      gridHeight=gridHeight,
      learningRate=0.0,
      log=""
    ) %>% 
    somSetLearningParameters() %>% 
    somlog(x, "SOM initiated: Input Size=", inputSize, ", Grid=", gridWidth,"*",gridHeight,"\n") %>% 
    somProgressFn()
  
  return(x)
}

somSetLearningParameters <- function(som, maxIterations=1000, startRate=0.8, endRate=0.003, startWidth=30, endWidth=5) {
  som$maxIterations=maxIterations
  som$startRate = startRate
  som$endRate = endRate
  som$startWidth = startWidth
  som$endWidth = endWidth
  return(som)
}


testsom = function() {
  samples=matrix(runif(15*3, min=-1.0, max=1.0),
    nrow=15,
    ncol=3)
  
  som=somInit(c(2,2,1), sd.method="nguyen.widrow")
  
  # som=somTrain(som, training, epochs=100, mini.batch.n=1, epochUpdateFreq=1, randomEpoch = F)
  som=somTrainStep(som, training)
  return(som)
  
  # x=somInit(c(2,2,1)) %>% 
  #   somTrain(training, epochs=100, epochUpdateFreq=1)
}




