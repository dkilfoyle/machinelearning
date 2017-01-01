library(shiny)
library(reshape2)
library(ggplot2)
library(dplyr)
library(visNetwork)
library(shinyBS)

source("dksom.R")
source("utils.R")

rValues = reactiveValues(MSE.df = data.frame(epoch=c(), MSE=c()),
  run.n=1, rsom=NULL)

ui <- fluidPage(
  
  shiny::tags$head(shiny::tags$style(shiny::HTML(
    "#consoleOutput { font-size: 11pt; height: 400px; overflow: auto; }"
  ))),
   
  # Application title
  titlePanel("Self Organizing Map"),
   
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
     
    sidebarPanel(
      includeCSS("styles.css"),
      
      selectInput("sDataset", "Dataset:", c("Colors","Dublin")),
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
  
  getTrainingData = reactive({
    isolate({
      if (input$sDataset == "Colors") {
        mydata =    matrix(runif(15*3, min=-1.0, max=1.0),
                           nrow=15,
                           ncol=3)
      }
      if (input$sDataset=="Dublin") {
        mydata = as.matrix(scale(readRDS("census.Rda")[,c(2,4,5,8)]))
      }
    })
    return(mydata)
  })
  
  getSom = reactive({
    if (is.null(rValues[["rsom"]])) {
      som = somInit(ncol(getTrainingData()), input$nGridWidth, input$nGridHeight) %>% 
        somSetLearningParameters(input$nMaxIterations, input$nStartRate, input$nEndRate, input$nStartWidth, input$nEndWidth) %>% 
        somProgressFn(function(x) progress$set(x))
      rValues$rsom = som
      return(som)
    }
    else
      return(rValues$rsom)
  })
  
  observeEvent(input$go, {
    
    progress = shiny::Progress$new(style="notification")
    progress$set(message="Training", value=0)
    on.exit(progress$close())
    
    som = getSom() %>% 
      somTrain(getTrainingData(), runName=input$txtRun)

    rValues$run.n = rValues$run.n + 1
    rValues$rsom = som
    
    updateTabsetPanel(session, "maintabs", "Plot")
    
  })
  
  observeEvent(input$go1, {
    som = getSom() %>% 
      somTrainStep(getTrainingData())
    
    rValues$rsom = som
    
    updateTabsetPanel(session, "maintabs", "Plot")
  })
  
  observeEvent(rValues$run.n, {
    updateTextInput(session, "txtRun", value=paste0("Run_",rValues$run.n))
  })
  
  observeEvent(input$btnClearRuns, {
    updateTabsetPanel(session, "maintabs", "Console")
    rValues$MSE.df = data.frame(epoch=c(), MSE=c())
    rValues$run.n=1
    rValues$rsom = NULL
  })
  
  output$consoleOutput = renderText({
    rValues$rsom$log
  })
  
 output$distPlot <- renderPlot({
   som = rValues$rsom
   multiplot(plotMeanBMU(som), plotNodeCount(som, getTrainingData()), plotsom(som), plotneighbor(som), cols=2)
 })
 
 output$network = renderVisNetwork({
   
 })
   
}

# Run the application 
shinyApp(ui = ui, server = server)

