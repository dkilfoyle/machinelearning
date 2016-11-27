library(shiny)
library(reshape2)
library(ggplot2)
library(dplyr)
library(visNetwork)

source("dkneuralnet2.R")

rValues = reactiveValues(MSE.df = data.frame(epoch=c(), MSE=c()),
  run.n=1)

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  shiny::tags$head(shiny::tags$style(shiny::HTML(
    "#consoleOutput { font-size: 11pt; height: 400px; overflow: auto; }"
  ))),
   
   # Application title
   titlePanel("NeuralNet"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
     
     sidebarPanel(
       radioButtons("rbDataset", "Dataset:", c("XOR","Titanic","Iris")),
       numericInput("nEpochs", "Epochs:", 500, min=1, max=10000, step=100),
       checkboxInput("bRandomEpoch","Randomize order each epoch: ", value=T),
       numericInput("nBatchSize", "Batch Size %:", 100, min=1, max=100),
       numericInput("nHidden1","Layer2 Hidden Neurons:", 2, min=0, max=100),
       numericInput("nHidden2","Layer3 Hidden Neurons:", 0, min=0, max=100),       
       radioButtons("rbWeightSD", "Weight Initiation SD:", c("sd=1.0","sd=1/sqrt(n)"),selected="sd=1/sqrt(n)"),
       selectInput("sMethod","Back Propogation Method:",c("Standard","RPROP"), selected="RPROP"),
       sliderInput("nTraining","Training Rate:", 0.7, min=0, max=1, step=0.1),
       sliderInput("nMomentum","Momentum:", 0.4, min=0, max=1, step=0.1),
       textInput("txtRun","Run Name:", "Run_1"),
       actionButton("btnClearRuns","Clear Runs"),
       actionButton("go1","Step 1"),
       actionButton("go","Go!")
     ),
      
      # Show a plot of the generated distribution
      mainPanel(
        tabsetPanel(
          tabPanel("Console", pre(id = "consoleOutput", class="shiny-text-output"), style="height:400px; margin-top:20px"), #verbatimTextOutput("console")),
          tabPanel("Plot", plotOutput("distPlot")),
          tabPanel("Network",
            radioButtons("rbVisEdges","Edges:", c("weights","biases","updateValues.w","lastWtChanges.w","lastWtChanges.b",
              "nabla_sum.w")),
            visNetworkOutput("network"))
        )
      )
   )
)

# Define server logic required to draw a histogram
server <- function(session, input, output) {
  
  getTrainedNetwork = eventReactive(input$go, {
    training=list()
    training[[1]]=list(c(0,0),c(0))
    training[[2]]=list(c(1,0),c(1))
    training[[3]]=list(c(0,1),c(1))
    training[[4]]=list(c(1,1),c(0))
    
    progress = shiny::Progress$new(style="notification")
    progress$set(message="Training", value=0)
    on.exit(progress$close())
    
    if (input$rbWeightSD == "sd=1/sqrt(n)")
      sd.method = "sqrtn"
    else
      sd.method = "1.0"
    
    if (input$nHidden2 > 0)
      sizes = c(2, input$nHidden1, input$nHidden2, 1)
    else
      sizes = c(2, input$nHidden1, 1)
    
    net = netInit(sizes, sd.method=sd.method) %>% 
      netProgressFn(function(x) progress$set(x))
    
    if (input$sMethod == "Standard") {
      net = net %>% 
        netStandardGradientDescent(eta=input$nTraining, momentum=input$nMomentum)
    }
    else if (input$sMethod=="RPROP") {
      net = net %>% 
        netRPROPGradientDescent()
    }
    
    net = net %>% 
      netTrain(training, epochs=input$nEpochs, mini.batch.size=input$nBatchSize, randomEpoch=input$bRandomEpoch)

    # MSE = SGD(training, input$nEpochs, 
    #   input$bRandomEpoch, 
    #   (input$nBatchSize/100)*length(training),
    #   input$sMethod,
    #   input$nTraining,
    #   input$nMomentum,
    #   training,
    #   progressFn=function(x) progress$set(x)
    # )
    
    rValues$run.n = rValues$run.n + 1
    
    return(net)
  })
  
  observeEvent(rValues$run.n, {
    updateTextInput(session, "txtRun", value=paste0("Run_",rValues$run.n))
  })
  
  observeEvent(input$btnClearRuns, {
    rValues$MSE.df = data.frame(epoch=c(), MSE=c())
    rValues$run.n=1
  })
  
  output$consoleOutput = renderPrint({
    getTrainedNetwork()
  })
  
 output$distPlot <- renderPlot({
   net = getTrainedNetwork()
   isolate({
     rValues$MSE.df = rbind(rValues$MSE.df, data.frame(epoch=1:input$nEpochs, MSE=net$MSE, run=input$txtRun))
     rValues$MSE.df %>%
       ggplot(aes(x=epoch, y=MSE, colour=run)) +
        geom_line() +
        ylim(0,1)
   })

   # 
 })
 
 output$network = renderVisNetwork({
   
   net=getTrainedNetwork()
   
   nodes = data.frame(id = 1:sum(net$sizes))
   edges = data.frame()
   
   nid = 1
   for (l in 1:net$num.layers) {
     for (n in 1:net$sizes[l]) {
       
       nodes$id[nid] = paste0("L",l,"N",n)
       
       if (l==1)
         nodes$label[nid] = paste0("I",n)
       else if (l==net$num.layers)
         nodes$label[nid] = paste0("O",n)
       else
         nodes$label[nid] = paste0("H",n)
       
       nodes$level[nid] = l
       nodes$shape="circle"
       nodes$value = net$activations[[l]][n]
       nodes$title = round(net$activations[[l]][n],2)
       
       nid = nid + 1
     }
   }
   
   eid = 1
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
           label = round(net[[input$rbVisEdges]][[l]][n,nprev],2),
           value = round(net[[input$rbVisEdges]][[l]][n,nprev],2),
           arrows = "to",
           color = rgb(colorRamp(c("blue","red"))(wc),max=255)
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

