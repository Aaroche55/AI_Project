# app.R

library(shiny)
library(caret)
library(randomForest)
library(hunspell)
library(stringr)
library(ggplot2)       
library(pROC)          
library(corrplot)      
library(dplyr) 

# Load the pre-trained model and pre-processing object
rf_tuned <- readRDS("rf_tuned_model.rds")
preProcValues <- readRDS("preProcValues.rds")

# Define feature calculation functions
calculate_num_chars <- function(text) {
  nchar(text)
}

calculate_burstiness <- function(text) {
  sentences <- unlist(strsplit(text, split = "[\\.!?]+"))
  sentences <- sentences[nchar(sentences) > 0]
  if (length(sentences) < 2) return(0)
  sentence_lengths <- sapply(sentences, nchar)
  sd(sentence_lengths) / mean(sentence_lengths)
}

calculate_lexical_diversity <- function(text) {
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[words != ""]
  length(unique(words)) / length(words)
}

calculate_misspelling_rate <- function(text) {
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[words != ""]
  misspelled <- sum(!hunspell_check(words))
  misspelled / length(words)
}

calculate_zipf_adherence <- function(text) {
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[words != ""]
  freq <- table(words)
  freq <- sort(freq, decreasing = TRUE)
  ranks <- 1:length(freq)
  log_ranks <- log(ranks)
  log_freq <- log(as.numeric(freq))
  abs(cor(log_ranks, log_freq))
}

calculate_entropy <- function(text) {
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[words != ""]
  freq <- table(words)
  probs <- freq / sum(freq)
  -sum(probs * log2(probs))
}

calculate_phrase_repetition <- function(text, n = 2) {
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[words != ""]
  if (length(words) < n) return(0)
  ngrams <- sapply(1:(length(words) - n + 1),
                   function(i) paste(words[i:(i + n - 1)], collapse = " "))
  sum(duplicated(ngrams)) / length(ngrams)
}

calculate_ai_words_rate <- function(text) {
  # Define a list of words that may be more common in AI-generated texts.
  ai_words <- c("algorithm", "model", "data", "predict", "machine",
                "learning", "artificial", "intelligence")
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[words != ""]
  sum(words %in% ai_words) / length(words)
}

# Function to compute all features for a given text
compute_features <- function(text) {
  data.frame(
    num_chars = calculate_num_chars(text),
    burstiness = calculate_burstiness(text),
    lexical_diversity = calculate_lexical_diversity(text),
    misspelling_rate = calculate_misspelling_rate(text),
    zipf_adherence = calculate_zipf_adherence(text),
    entropy = calculate_entropy(text),
    phrase_repetition = calculate_phrase_repetition(text),
    ai_words_rate = calculate_ai_words_rate(text)
  )
}

# Prediction function
predict_text <- function(text, preProcValues, model) {
  # Compute features for the input text
  new_data <- compute_features(text)
  
  # Apply the same scaling/preprocessing as in training
  new_data_scaled <- predict(preProcValues, newdata = new_data)
  
  # Predict probabilities using the trained model
  prediction_prob <- predict(model, newdata = new_data_scaled, type = "prob")
  
  # Extract probability for the "AI" class (assuming "AI" is the positive class label)
  ai_prob <- prediction_prob[, "AI"]
  
  # Determine predicted class based on a 0.5 cutoff
  predicted_class <- ifelse(ai_prob > 0.5, "AI", "Human")
  
  list(
    prediction = predicted_class,
    confidence = ai_prob
  )
}

# Define UI
ui <- fluidPage(
  titlePanel("Essay Classification: AI vs Human"),
  sidebarLayout(
    sidebarPanel(
      textAreaInput("inputText", "Enter text:", 
                    rows = 15, 
                    placeholder = "Type or paste your text here..."),
      actionButton("predictButton", "Predict")
    ),
    mainPanel(
      h3("Prediction Result"),
      verbatimTextOutput("resultText")
    )
  )
)

# Define Server Logic
server <- function(input, output, session) {
  result <- eventReactive(input$predictButton, {
    req(input$inputText)  # Ensure input is not empty
    predict_text(input$inputText, preProcValues, rf_tuned)
  })
  
  output$resultText <- renderPrint({
    if (is.null(result())) return("Waiting for input...")
    cat("Predicted Class:", result()$prediction, "\n")
    cat("Probability of AI:", round(result()$confidence, 3), "\n")
  })
}

# Run the app
shinyApp(ui = ui, server = server)
