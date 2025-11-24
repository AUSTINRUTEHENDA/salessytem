#############################################
#  RETAIL SALES PREDICTION SYSTEM (Shiny)   #
#############################################

# 0. LOAD PACKAGES -------------------------------------------------------

library(shiny)
library(randomForest)
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)

# 1. LOAD YOUR SALES DATA -----------------------------------------------

data_file <- "sales_data.csv"

if (!file.exists(data_file)) {
  stop(
    paste0(
      "\n⚠ 'sales_data.csv' was NOT found in folder: ", getwd(), "\n",
      "Create a CSV file named 'sales_data.csv' with columns:\n",
      "shop_id,date,sales,promotion,holiday,weather\n\n",
      "Example row:\n",
      "Shop1,2024-01-01,120,Yes,No,Sunny\n"
    )
  )
}

sales_data <- read_csv(data_file, show_col_types = FALSE)

# Check required columns
needed_cols <- c("shop_id", "date", "sales", "promotion", "holiday", "weather")
missing_cols <- setdiff(needed_cols, names(sales_data))

if (length(missing_cols) > 0) {
  stop(paste("These required columns are missing in sales_data.csv:",
             paste(missing_cols, collapse = ", ")))
}

# 2. CLEAN & FEATURE ENGINEERING ----------------------------------------
# Use numeric day-of-week and month so future dates work.

sales_data <- sales_data %>%
  mutate(
    date      = as.Date(date),
    shop_id   = factor(shop_id),
    promotion = factor(promotion),
    holiday   = factor(holiday),
    weather   = factor(weather),
    dow_num   = lubridate::wday(date, week_start = 1),  # 1 = Mon ... 7 = Sun
    month_num = lubridate::month(date)                  # 1..12
  ) %>%
  filter(!is.na(sales))

# 3. TRAIN / TEST SPLIT --------------------------------------------------

set.seed(123)

n <- nrow(sales_data)
train_idx <- sample(seq_len(n), size = floor(0.8 * n))
train_data <- sales_data[train_idx, ]
test_data  <- sales_data[-train_idx, ]

# 4. TRAIN RANDOM FOREST MODEL ------------------------------------------

rf_formula <- sales ~ shop_id + dow_num + month_num + promotion + holiday + weather

rf_model <- randomForest(
  formula    = rf_formula,
  data       = train_data,
  ntree      = 500,
  mtry       = 3,
  importance = TRUE
)

# 5. OPTIONAL: EVALUATE ON TEST SET (prints in console) ------------------

if (nrow(test_data) > 0) {
  test_pred <- predict(rf_model, newdata = test_data)
  rmse <- sqrt(mean((test_pred - test_data$sales)^2))
  message("Test RMSE: ", round(rmse, 2))
}

# 6. SHINY APP UI -------------------------------------------------------

ui <- fluidPage(
  titlePanel("Retail Sales Prediction System"),
  
  sidebarLayout(
    sidebarPanel(
      dateInput("date", "Select date:", value = Sys.Date()),
      selectInput(
        "shop_id", "Shop:",
        choices = sort(unique(sales_data$shop_id))
      ),
      selectInput(
        "promotion", "Promotion active?",
        choices = sort(unique(as.character(sales_data$promotion)))
      ),
      selectInput(
        "holiday", "Holiday?",
        choices = sort(unique(as.character(sales_data$holiday)))
      ),
      selectInput(
        "weather", "Weather:",
        choices = sort(unique(as.character(sales_data$weather)))
      ),
      actionButton("predict_btn", "Predict Sales")
    ),
    
    mainPanel(
      h3("Predicted Sales"),
      verbatimTextOutput("prediction_text"),
      hr(),
      h3("Historical Sales Trend (Selected Shop)"),
      plotOutput("trend_plot")
    )
  )
)

# 7. SHINY SERVER -------------------------------------------------------

server <- function(input, output, session) {
  
  pred_result <- eventReactive(input$predict_btn, {
    req(input$date, input$shop_id)
    
    new_date <- as.Date(input$date)
    
    new_data <- data.frame(
      shop_id   = factor(input$shop_id, levels = levels(sales_data$shop_id)),
      promotion = factor(input$promotion, levels = levels(sales_data$promotion)),
      holiday   = factor(input$holiday,   levels = levels(sales_data$holiday)),
      weather   = factor(input$weather,   levels = levels(sales_data$weather)),
      dow_num   = lubridate::wday(new_date, week_start = 1),
      month_num = lubridate::month(new_date)
    )
    
    pred_value <- as.numeric(predict(rf_model, newdata = new_data))
    
    data.frame(
      date       = new_date,
      shop_id    = input$shop_id,
      prediction = pred_value
    )
  })
  
  output$prediction_text <- renderText({
    req(pred_result())
    p <- pred_result()
    
    if (is.na(p$prediction)) {
      return("Prediction not available for this combination (insufficient training data).")
    }
    
    paste0(
      "Predicted sales for shop ", p$shop_id,
      " on ", format(p$date, "%Y-%m-%d"),
      " = ", round(p$prediction, 2), " units."
    )
  })
  
  output$trend_plot <- renderPlot({
    req(input$shop_id)
    
    sales_data %>%
      filter(shop_id == input$shop_id) %>%
      ggplot(aes(x = date, y = sales)) +
      geom_line() +
      geom_point() +
      labs(
        x = "Date",
        y = "Sales",
        title = paste("Historical Sales for Shop", input$shop_id)
      )
  })
}

# 8. RUN APP ------------------------------------------------------------

shinyApp(ui, server)
