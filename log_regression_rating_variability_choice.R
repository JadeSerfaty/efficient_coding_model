library(lme4)
library(MASS)  # For random data generation and logistic function

# Define emotions
emotions <- c("joy", "sadness", "anxiety", "romance", "disgust")
# 
# # Define number of simulations and range of sample sizes
# n_sim <- 1000
# sample_sizes <- seq(30, 250, by = 10)
# 
# # Function to perform simulations and check significance
# perform_simulations <- function(meanVdiff, sd_Vdiff, mean_Var, sd_Var, mean_Svalue, sd_Svalue, b0, b1, b2, b3, sample_sizes, n_sim) {
#   success_rates <- numeric(length(sample_sizes))
#   
#   for (i in seq_along(sample_sizes)) {
#     n <- sample_sizes[i]
#     significant_count <- 0
#     
#     for (j in 1:n_sim) {
#       
#       # Then in your simulation function, you might use:
#       Vdiff <- rnorm(n, mean_Vdiff, sd_Vdiff)  # Assuming normal distribution
#       Var <- rnorm(n, mean_Var, sd_Var)        # Assuming normal distribution
#       Svalue <- rnorm(n, mean_Svalue, sd_Svalue) # Assuming normal distribution
#       
#       # Calculate the probability using the logistic function
#       logits <- b0 + b1 * Vdiff + b2 * Var + b3 * Svalue
#       probability <- plogis(logits)
#       
#       # Generate binary outcomes based on the calculated probabilities
#       Choice_true <- rbinom(n, 1, probability)
#       
#       # Analysis (assuming you know b1 and its standard error from pilot study fitting)
#       SE_b1 <- sqrt(var(Vdiff) / (n * var(probability) * (1 - var(probability))))
#       z_value <- b1 / SE_b1
#       
#       # Check if b1 is statistically significant
#       if (abs(z_value) > qnorm(0.975)) {  # Using normal approximation for z-test
#         significant_count <- significant_count + 1
#       }
#     }
#     
#     success_rates[i] <- significant_count / n_sim
#   }
#   
#   return(success_rates)
# }
# 
# # Loop through each emotion to load data, fit the model, and perform simulations
# results <- list()

for (emotion in emotions) {
  filename <- paste("/Users/jadeserfaty/Library/Mobile Documents/com~apple~CloudDocs/code/lido/introspection_task/main_study_data/main_study_choice_data/choice_data_", emotion, ".csv", sep="")
  data <- read.csv(filename)
  
  # Handle NA data
  # complete_data <- na.omit(data)
  
  # mean_Vdiff <- mean(complete_data$value_difference, na.rm = TRUE)
  # sd_Vdiff <- sd(complete_data$value_difference, na.rm = TRUE)
  # mean_Var <- mean(complete_data$summed_variability, na.rm = TRUE)
  # sd_Var <- sd(complete_data$summed_variability, na.rm = TRUE)
  # mean_Svalue <- mean(complete_data$summed_value, na.rm = TRUE)
  # sd_Svalue <- sd(complete_data$summed_value, na.rm = TRUE)

  # Fit the mixed-effects logistic regression model
  model <- glmer(consistent_choice ~ value_difference + summed_variability + summed_value + (1 | subject_id), 
                 data = data, 
                 family = binomial(link = "logit"))
  print(emotion)
  print(summary(model))
  
  # # Extract coefficients
  # coef_est <- fixef(model)
  # b0 <- coef_est[1]  # Intercept
  # b1 <- coef_est[2]  # Coefficient for value_difference
  # b2 <- coef_est[3]  # Coefficient for summed_variability
  # b3 <- coef_est[4]  # Coefficient for summed_value
  
  # # Perform simulations
  # emotion_results <- perform_simulations(meanVdiff, sd_Vdiff, mean_Var, sd_Var, mean_Svalue, sd_Svalue, b0, b1, b2, b3, sample_sizes, n_sim)
  # results[[emotion]] <- emotion_results
  # 
  # # Optionally find the minimum sample size for desired power level, e.g., 90%
  # min_sample_size <- sample_sizes[min(which(emotion_results >= 0.9))]
  # cat(sprintf("Minimum sample size for %s for 90%% power: %d\n", emotion, min_sample_size))
}

# # Save or further process results
# print(results)