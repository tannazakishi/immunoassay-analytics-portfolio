# ==============================================================================
# Script Name: linearity_analysis.R
# Description: Demonstrates linear regression and linearity diagnostic analysis.
# Author: Tannaz Akishi

# ==============================================================================

# Load required libraries
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("broom", quietly = TRUE)) install.packages("broom")

library(ggplot2)
library(broom)

# ------------------------------------------------------------------------------
# 1. Generate Synthetic Calibration Data
# ------------------------------------------------------------------------------
set.seed(42)

# Simulating standard analytical concentration levels (e.g., 5 levels, 3 replicates each)
concentrations <- rep(c(10, 20, 50, 100, 200), each = 3)
slope <- 1.5
intercept <- 2.0
noise <- rnorm(length(concentrations), mean = 0, sd = 3.5)

# Calculate simulated instrumental signal
signal <- intercept + slope * concentrations + noise

data <- data.frame(
  Concentration = concentrations,
  Signal = signal
)

# ------------------------------------------------------------------------------
# 2. Fit Simple Linear Regression
# ------------------------------------------------------------------------------
linear_model <- lm(Signal ~ Concentration, data = data)
model_summary <- summary(linear_model)

cat("==========================================\n")
cat("      LINEAR REGRESSION MODEL RESULTS     \n")
cat("==========================================\n")
print(model_summary)

r_squared <- model_summary$r.squared
cat(sprintf("\nCoefficient of Determination (R²): %.4f\n", r_squared))

# ------------------------------------------------------------------------------
# 3. Lack-of-Fit Test (ANOVA)
# ------------------------------------------------------------------------------
# Fit full model where concentration is treated as a categorical factor
factor_model <- lm(Signal ~ factor(Concentration), data = data)

# Perform Analysis of Variance comparing linear vs. factor model
lof_test <- anova(linear_model, factor_model)

cat("\n==========================================\n")
cat("          LACK-OF-FIT ANOVA TEST          \n")
cat("==========================================\n")
print(lof_test)

# ------------------------------------------------------------------------------
# 4. Residual Diagnostics
# ------------------------------------------------------------------------------
data_diagnostics <- augment(linear_model)

# Plot 1: Calibration Curve with Linear Fit
p1 <- ggplot(data, aes(x = Concentration, y = Signal)) +
  geom_point(color = "#1F77B4", size = 3, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "#D62728", fill = "#FF9E9E") +
  labs(
    title = "Calibration Curve & Linearity Fit",
    subtitle = paste("R² =", round(r_squared, 4)),
    x = "Concentration (mg/L)",
    y = "Signal Response (AU)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# Plot 2: Residuals vs. Fitted Values
p2 <- ggplot(data_diagnostics, aes(x = .fitted, y = .resid)) +
  geom_point(color = "#1F77B4", size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  geom_smooth(method = "loess", se = FALSE, color = "#2CA02C", linetype = "dotted") +
  labs(
    title = "Residuals vs Fitted",
    subtitle = "Check for homoscedasticity and non-linear patterns",
    x = "Fitted Values",
    y = "Residuals"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# Save plots to disk
ggsave("calibration_curve.png", plot = p1, width = 6, height = 4.5, dpi = 300)
ggsave("residual_plot.png", plot = p2, width = 6, height = 4.5, dpi = 300)

cat("\nGraphics successfully saved to working directory.\n")
