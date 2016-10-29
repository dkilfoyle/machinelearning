library(shiny)
library(reshape2)
library(ggplot2)
library(dplyr)
library(visNetwork)

source("dkneuralnet.R")

rValues = reactiveValues(MSE.df = data.frame(epoch=c(), MSE=c()))

# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("NeuralNet"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
     
     sidebarPanel(
       radioButtons("rbDataset", "Dataset:", c("XOR","Titanic","Iris")),
       numericInput("nEpochs", "Epochs:", 500, min=1, max=10000, step=100),
       numericInput("nBatchSize", "Batch Size %:", 100, min=1, max=100),
       numericInput("nHidden","Number Hidden Neurons:", 2, min=0, max=100),
       numericInput("nTraining","Training Rate:", 0.7, min=0, max=1, step=0.1),
       numericInput("nMomentum","Momentum:", 0.1, min=0, max=1, step=0.1),
       textInput("tRun","Run Name:", "Run1"),
       actionButton("go","Go!")
     ),
      
      # Show a plot of the generated distribution
      mainPanel(
        tabsetPanel(
          tabPanel("Console", verbatimTextOutput("console")),
          tabPanel("Plot", plotOutput("distPlot")),
          tabPanel("Network", visNetworkOutput("network"))
        )
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  getMSE = reactive({
    return(MSE.df)
  })
   
  getTrainedNetwork = eventReactive(input$go, {
    training=list()
    training[[1]]=list(c(0,0),c(0))
    training[[2]]=list(c(1,0),c(1))
    training[[3]]=list(c(0,1),c(1))
    training[[4]]=list(c(1,1),c(0))
    
    initNetwork(c(2,input$nHidden,1))
    
    progress = shiny::Progress$new(style="notification")
    progress$set(message="Training", value=0)
    on.exit(progress$close())
    
    MSE = SGD(training, input$nEpochs, 
      (input$nBatchSize/100)*length(training),
      input$nTraining,
      input$nMomentum,
      training,
      progressFn=function(x) progress$set(x)
    )
    rValues$MSE.df = rbind(rValues$MSE.df, data.frame(epoch=1:input$nEpochs, MSE=MSE, run=input$tRun))
  })
  
  output$console = renderPrint({
    getTrainedNetwork()
  })
  
 output$distPlot <- renderPlot({
   rValues$MSE.df %>%
     # melt(id="epoch") %>%
     ggplot(
       aes(x=epoch, y=MSE, colour=run)) +
     geom_line()
   # 
 })
 
 output$network = renderVisNetwork({
   
   getTrainedNetwork()
   
   nodes = data.frame(id = 1:sum(sizes))
   edges = data.frame()
   
   nid = 1
   for (l in 1:num.layers) {
     for (n in 1:sizes[l]) {
       
       nodes$id[nid] = paste0("L",l,"N",n)
       
       if (l==1)
         nodes$label[nid] = paste0("I",n)
       else if (l==num.layers)
         nodes$label[nid] = paste0("O",n)
       else
         nodes$label[nid] = paste0("H",n)
       
       nodes$level[nid] = l
       nodes$shape="circle"
       nodes$value = activations[[l]][n]
       nodes$title = round(activations[[l]][n],2)
       
       nid = nid + 1
     }
   }
   
   eid = 1
   for (l in num.layers:2) {
     for (n in 1:sizes[l]) {
       for (nprev in 1:sizes[l-1]) {
         
         # convert weight into range 0..1
         ws = max(abs(weights[[l]]))
         w = weights[[l]][n,nprev]
         wc = max(w,-1)
         wc = min(wc,1)
         wc = (w + ws)/(ws*2)
         
         edges = rbind(edges, data.frame(
           from = paste0("L",l-1,"N",nprev),
           to = paste0("L",l,"N",n),
           label = round(weights[[l]][n,nprev],2),
           value = round(weights[[l]][n,nprev],2),
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

