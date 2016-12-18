library(shiny)
library(reshape2)
library(ggplot2)
library(dplyr)
library(visNetwork)

source("dkneuralnet2.R")

rValues = reactiveValues(MSE.df = data.frame(epoch=c(), MSE=c()),
  run.n=1, rnet=list())

ui <- fluidPage(
  
  shiny::tags$head(shiny::tags$style(shiny::HTML(
    "#consoleOutput { font-size: 11pt; height: 400px; overflow: auto; }"
  ))),
   
  # Application title
  titlePanel("NeuralNet"),
   
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
     
    sidebarPanel(
      selectInput("rbDataset", "Dataset:", c("XOR","Titanic","Iris")),
      numericInput("nEpochs", "Max Epochs:", 500, min=1, max=10000, step=100),
      numericInput("nMaxError","Max Error:", 0.01, min=0.0, step=0.01),
      radioButtons("rbBatchMethod", "Batch Method:", c("Online","Batch")),
      conditionalPanel(condition="input.rbBatchMethod=='Batch'",
        checkboxInput("bRandomEpoch","Randomize order each epoch: ", value=F),
        numericInput("nBatchSize", "Batch Size %:", 100, min=1, max=100)),
      numericInput("nHidden1","Layer2 Hidden Neurons:", 2, min=0, max=100),
      numericInput("nHidden2","Layer3 Hidden Neurons:", 0, min=0, max=100),       
      selectInput("rbWeightSD", "Weight Initiation SD:", c("sd=1.0","sd=1/sqrt(n)","nguyen.widrow", "aifh.xor"),selected="sd=1.0"),
      selectInput("sMethod","Back Propogation Method:",c("Standard","RPROP"), selected="Standard"),
      sliderInput("nTraining","Training Rate:", 0.7, min=0, max=1, step=0.1),
      sliderInput("nMomentum","Momentum:", 0.3, min=0, max=1, step=0.1),
      textInput("txtRun","Run Name:", "Run_1"),
      actionButton("btnClearRuns","Clear Runs"),
      actionButton("go1","Step 1"),
      actionButton("go","Go!")
    ),
      
    # Show a plot of the generated distribution
    mainPanel(
      tabsetPanel(
        tabPanel("Info",
          withMathJax(includeHTML("math.html"))),
        tabPanel("Console", pre(id = "consoleOutput", class="shiny-text-output"), style="height:400px; margin-top:20px"), #verbatimTextOutput("console")),
        tabPanel("Plot", plotOutput("distPlot"), style="margin-top:20px"),
        tabPanel("Network", 
          fluidRow(
            column(6,
              radioButtons("rbVisEdges","Edges:", c("weights","updateValues.w","gradient.w", "gradient_sum.w", "lastWtChanges.w"))
            ),
            column(6,
              radioButtons("rbVisNodes","Nodes:", c("z","activations","delta"))
            )
          ),
          visNetworkOutput("network"),  style="margin-top:20px"),
      id="maintabs")
    )
  )
)

server <- function(session, input, output) {
  
  getTrainingData = function() {
    training=list()
    training[[1]]=list(c(0,0),c(0))
    training[[2]]=list(c(1,0),c(1))
    training[[3]]=list(c(0,1),c(1))
    training[[4]]=list(c(1,1),c(0))
    return(training)
  }
  
  initNetwork = function() {
    if (input$rbWeightSD == "sd=1/sqrt(n)")
      sd.method = "sqrtn"
    else
      sd.method = "1.0"
    
    # TODO: Calculate input and output layer neuron size from training data
    if (input$nHidden2 > 0)
      sizes = c(2, input$nHidden1, input$nHidden2, 1)
    else
      sizes = c(2, input$nHidden1, 1)
    
    net = netInit(sizes, sd.method=sd.method)

    if (input$sMethod == "Standard") {
      net = net %>% 
        netStandardGradientDescent(eta=input$nTraining, momentum=input$nMomentum)
    }
    else if (input$sMethod=="RPROP") {
      net = net %>% 
        netRPROPGradientDescent()
    }
    
    # initial weights to mimic example http://www.heatonresearch.com/aifh/vol3/xor_online.html
    
    if (input$rbWeightSD == "aifh.xor") {
      # H1 receiving weights
      net$weights[[2]][1,1] = -0.06782947598673161
      net$weights[[2]][1,2] =  0.22341077197888182
      net$biases[[2]][1] = -0.4635107399577998
      
      # H2 receiving weights
      net$weights[[2]][2,1] =  0.9487814395569221
      net$weights[[2]][2,2] =  0.46158711646254
      net$biases[[2]][2] =    0.09750161997450091
      
      # o1 receiving weights
      net$weights[[3]][1,1] = -0.22791948943117624
      net$weights[[3]][1,2] =  0.581714099641357
      net$biases[[3]][1] =    0.7792991203673414
    }

    return(net)
  }
  
  observeEvent(input$go, {
    
    progress = shiny::Progress$new(style="notification")
    progress$set(message="Training", value=0)
    on.exit(progress$close())
    
    net = initNetwork() %>% 
      netProgressFn(function(x) progress$set(x)) %>% 
      netTrain(getTrainingData(), 
        maxepochs=input$nEpochs,
        maxerror=input$nMaxError,
        mini.batch.percent=input$nBatchSize,
        randomEpoch=input$bRandomEpoch,
        test.data=getTrainingData())
    
    rValues$run.n = rValues$run.n + 1
    rValues$rnet = net
    
    updateTabsetPanel(session, "maintabs", "Plot")
    
  })
  
  observeEvent(input$go1, {
    if (is.null(rValues$rnet$step))
      net = initNetwork()
    else
      net = rValues$rnet
    
    net = netTrainStep(net, getTrainingData(), input$rbBatchMethod)
    
    rValues$rnet = net
    
    updateTabsetPanel(session, "maintabs", "Console")
  })
  
  observeEvent(rValues$run.n, {
    updateTextInput(session, "txtRun", value=paste0("Run_",rValues$run.n))
  })
  
  observeEvent(input$btnClearRuns, {
    rValues$MSE.df = data.frame(epoch=c(), MSE=c())
    rValues$run.n=1
    rValues$rnet = list()
  })
  
  output$consoleOutput = renderText({
    rValues$rnet$log
  })
  
 output$distPlot <- renderPlot({
   net = rValues$rnet
   
   isolate({
     # rValues$MSE.df = rbind(rValues$MSE.df, data.frame(epoch=1:input$nEpochs, MSE=net$MSE, run=input$txtRun))
     rValues$MSE.df = data.frame(epoch=1:length(net$MSE), MSE=net$MSE, run=input$txtRun)
     
     rValues$MSE.df %>%
       ggplot(aes(x=epoch, y=MSE, colour=run)) +
        geom_line() +
        ylim(0,1)
   })

   # 
 })
 
 output$network = renderVisNetwork({
   
   net=rValues$rnet
   
   nodes = data.frame() 
   edges = data.frame()
   
   for (l in 1:net$num.layers) {
     for (n in 1:net$sizes[l]) {
       
       if (l==1)
         label = paste0("I",n)
       else if (l==net$num.layers)
         label = paste0("O",n)
       else
         label = paste0("H",n)
       
       value = net[[input$rbVisNodes]][[l]][n]
       if (is.null(value)) {
         title = 0
         value = 0
       }
       else
       {
         title = round(value,3)
       }
       
       nodes = rbind(nodes, data.frame(
         id = paste0("L",l,"N",n),
         label = label,
         level = l,
         shape = "circle",
         value = value,
         title = title
       ))
     }
       
   if (l > 1)
     nodes = rbind(nodes, data.frame(
       id = paste0("L",l-1,"B"),
       label = paste0("B",l),
       level=l,
       shape="circle",
       value = 1,
       title = 1
     ))
   }
   
   for (l in net$num.layers:2) {
     for (n in 1:net$sizes[l]) {
       for (nprev in 1:net$sizes[l-1]) {
         
         # convert weight into range 0..1
         ws = max(abs(net$weights[[l]]))
         w = net$weights[[l]][n,nprev]
         wc = max(w,-1)
         wc = min(wc,1)
         wc = (w + ws)/(ws*2)
         
         edges = rbind(edges, data.frame(
           from = paste0("L",l-1,"N",nprev),
           to = paste0("L",l,"N",n),
           label = round(net[[input$rbVisEdges]][[l]][n,nprev],3),
           value = round(net[[input$rbVisEdges]][[l]][n,nprev],3),
           arrows = "to",
           color = rgb(colorRamp(c("blue","red"))(wc),max=255)
         ))
           
       }
       
       if (l > 1) {
         if (input$rbVisEdges == "weights")
           value = net$biases[[l]][n]
         else if (input$rbVisEdges == "updateValues.w")
           value = net$updateValues.b[[l]][n]
         else if (input$rbVisEdges == "gradient.w")
           value = net$gradient.b[[l]][n]
         else if (input$rbVisEdges == "gradient_sum.w")
           value = net$gradient_sum.b[[l]][n]
         else if (input$rbVisEdges == "lastWtChanges.w")
           value = net$lastWtChanges.b[[l]][n]
         edges = rbind(edges, data.frame(
           from = paste0("L",l-1,"B"),
           to = paste0("L",l,"N",n),
           label=round(value,3),
           value=value,
           arrows="to",
           color = rgb(0.5,0.5,0.5)
         ))
       }
     }
   }
   
   visNetwork(nodes, edges) %>% 
     visHierarchicalLayout(direction = "LR") %>%
     visEdges(scaling=list(max=5, label=list(enabled=F))) %>%  #TODO: customs scaling function for negative values
     visNodes(scaling=list(max=5)) %>% 
     visInteraction(hover = TRUE)
 })
   
   
}

# Run the application 
shinyApp(ui = ui, server = server)

