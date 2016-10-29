library(shiny)
library(reshape2)
library(ggplot2)
library(dplyr)

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
          tabPanel("Network", plotOutput("network"))
        )
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  getMSE = reactive({
    return(MSE.df)
  })
   
  do.net = eventReactive(input$go, {
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
    do.net()
  })
  
   output$distPlot <- renderPlot({
     rValues$MSE.df %>%
       # melt(id="epoch") %>%
       ggplot(
         aes(x=epoch, y=MSE, colour=run)) +
       geom_line()
     # 
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

