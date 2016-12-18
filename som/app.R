library(shiny)
library(reshape2)
library(ggplot2)
library(dplyr)
library(visNetwork)
library(shinyBS)

source("dksom.R")

rValues = reactiveValues(MSE.df = data.frame(epoch=c(), MSE=c()),
  run.n=1, rnet=list())

ui <- fluidPage(
  
  shiny::tags$head(shiny::tags$style(shiny::HTML(
    "#consoleOutput { font-size: 11pt; height: 400px; overflow: auto; }"
  ))),
   
  # Application title
  titlePanel("Self Organizing Map"),
   
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
     
    sidebarPanel(
      selectInput("rbDataset", "Dataset:", c("Colors","Titanic","Iris")),
      numericInput("nMaxIterations", "Max Iterations:", 500, min=1, max=10000, step=100),
      
      bsCollapse(id="Options",
        bsCollapsePanel("Grid",
          numericInput("nGridWidth","Width:", 50, min=5, step=1),
          numericInput("nGridHeight", "Height:", 50, min=5, step=1)),
        bsCollapsePanel("Learning",
          sliderInput("nStartRate","Start Training Rate:", 0.8, min=0.1, max=1, step=0.1),
          numericInput("nEndRate","End Training Rate:", 0.003, min=0.0, max=1, step=0.0005), 
          sliderInput("nStartWidth","Start Neighbour Width:", 30, min=1, max=100, step=1),
          sliderInput("nEndWidth","End Neighbour Width:", 5, min=1, max=100, step=1)
          )),

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
    samples=matrix(runif(15*3, min=-1.0, max=1.0),
      nrow=15,
      ncol=3)
    return(samples)
  }
  
  initSom = function(samples) {
    som = somInit(ncol(samples), input$nGridWidth, input$nGridHeight) %>% 
      somSetLearningParameters(input$nMaxIterations, input$nStartRate, input$nEndRate, input$nStartWidth, input$nEndWidth) %>% 
      somProgressFn(function(x) progress$set(x))
    return(som)
  }
  
  observeEvent(input$go, {
    
    progress = shiny::Progress$new(style="notification")
    progress$set(message="Training", value=0)
    on.exit(progress$close())
    
    samples=getTrainingData()
    initSom(samples) %>% 
      somTrain(samples, runName=input$txtRun)

    rValues$run.n = rValues$run.n + 1
    rValues$rsom = som
    
    updateTabsetPanel(session, "maintabs", "Plot")
    
  })
  
  observeEvent(input$go1, {
    if (is.null(rValues$rsom$iteration))
      som = initNetwork()
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
     rValues$MSE.df = rbind(rValues$MSE.df, data.frame(epoch=1:length(net$MSE), MSE=net$MSE, run=net$runName))
     
     rValues$MSE.df %>%
       ggplot(aes(x=epoch, y=MSE, colour=run)) +
        geom_line() +
        ylim(0,0.5)
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

