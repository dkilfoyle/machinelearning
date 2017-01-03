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
      
      bsCollapse(id="Options",
        bsCollapsePanel("Training",
          numericInput("nMaxIterations", "Max Iterations:", 500, min=1, max=10000, step=100),
          numericInput("nEvaluateSize", "Evaluate Sample Size (%):", 100, min=1, max=100, step=10),
          numericInput("nEvaluateFrequency", "Evaluate Frequency (per N iterations):", 10, min=1, max=1000, step=10)),
        bsCollapsePanel("Grid",
          numericInput("nGridWidth","Width:", 50, min=5, step=5),
          numericInput("nGridHeight", "Height:", 50, min=5, step=5)),
        bsCollapsePanel("Learning",
          numericInput("nStartRate","Start Training Rate:", 0.8, min=0.1, max=1, step=0.1),
          numericInput("nEndRate","End Training Rate:", 0.003, min=0.0, max=1, step=0.0005), 
          sliderInput("nStartWidth","Start Neighbour Width (%):", 60, min=1, max=100, step=1),
          sliderInput("nEndWidth","End Neighbour Width (%):", 5, min=1, max=100, step=1)
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
        tabPanel("Plot", 
          fluidRow(
            column(6, selectInput("sFeature","Feature:", c("RGB","R","G","B"))),
            column(6, numericInput("nNumClusters","Number of Clusters", 5)),
            column(12, 
              plotOutput("distPlot"), style="margin-top:20px"))),
      id="maintabs")
    )
  )
)

server <- function(session, input, output) {
  
  observeEvent(input$sDataset, {
    if (input$sDataset=="Colors") {
      updateNumericInput(session, "nMaxIterations", value=500)
      updateNumericInput(session, "nEvaluateSize", value=100)
      updateNumericInput(session, "nGridWidth", value=50)
      updateNumericInput(session, "nGridHeight", value=50)
      updateNumericInput(session, "nStartRate", value=0.8)
      updateNumericInput(session, "nEndRate", value=0.003)
      updateSliderInput(session, "nStartWidth", value=60)
      updateSliderInput(session, "nEndWidth", value=5)
      rValues$rsom = NULL
      featureList = c("RGB")
    }
    if (input$sDataset=="Dublin") {
      updateNumericInput(session, "nMaxIterations", value=100)
      updateNumericInput(session, "nEvaluateSize", value=10)
      updateNumericInput(session, "nGridWidth", value=20)
      updateNumericInput(session, "nGridHeight", value=20)
      updateNumericInput(session, "nStartRate", value=0.5)
      updateNumericInput(session, "nEndRate", value=0.01)
      updateSliderInput(session, "nStartWidth", value=60)
      updateSliderInput(session, "nEndWidth", value=5)
      rValues$rsom = NULL
      featureList = c()
    }
    mydata = getTrainingData()
    updateSelectInput(session, "sFeature", "Feature:", c(featureList, colnames(mydata)))
  })
  
  getTrainingData = function() {
    isolate({
    if (input$sDataset == "Colors") {
      mydata =    matrix(runif(15*3, min=-1.0, max=1.0),
                         nrow=15,
                         ncol=3)
      dimnames(mydata) = list(NULL, c("R","G","B"))

    }
    if (input$sDataset=="Dublin") {
      mydata = as.matrix(scale(readRDS("census.Rda")[,c(2,4,5,8)]))
    }

    return(mydata)
    })
  }
  
  getSom = reactive({
    if (is.null(rValues[["rsom"]])) {
      som = somInit(inputSize = ncol(getTrainingData()), 
                    gridWidth = input$nGridWidth,
                    gridHeight = input$nGridHeight,
                    evaluateFrequency = input$nEvaluateFrequency,
                    evaluateSampleProp = input$nEvaluateSize/100) %>% 
        somSetLearningParameters(input$nMaxIterations, 
                                 input$nStartRate,
                                 input$nEndRate,
                                 max(1, floor(input$nStartWidth/100 * input$nGridWidth)),
                                 max(1, floor(input$nEndWidth/100 * input$nGridWidth)))

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
      somProgressFn(function(x) progress$set(x)) %>% 
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
   if (is.null(som)) return(NULL)
   
   if (som$stepping) {
     multiplot(plotBMUCount(som, getTrainingData()),
               plotNeighbor(som),
               plotSOMFeature(som,input$sFeature),
               plotClusters(som, input$nNumClusters), cols=2)
   }
   else {
     multiplot(plotMeanBMU(som), 
               plotBMUCount(som, getTrainingData()),
               plotSOMFeature(som,input$sFeature),
               plotClusters(som, input$nNumClusters), cols=2)
   }
   

 })
 
 output$network = renderVisNetwork({
   
 })
   
}

# Run the application 
shinyApp(ui = ui, server = server)

