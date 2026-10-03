# ==============================================================================
# Script Name: linearity_analysis.py
# Description: Demonstrates linear regression and linearity diagnostic analysis.
# Author: Tannaz Akishi (Python Translation)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ------------------------------------------------------------------------------
# 1. Generate Synthetic Calibration Data
# ------------------------------------------------------------------------------
np.random.seed(42)

# Simulating standard analytical concentration levels (5 levels, 3 replicates each)
concentrations = np.repeat([10, 20, 50, 100, 200], 3)
slope = 1.5
intercept = 2.0
noise = np.random.normal(loc=0, scale=3.5, size=len(concentrations))

# Calculate simulated instrumental signal
signal = intercept + slope * concentrations + noise

data = pd.DataFrame({
    'Concentration': concentrations,
    'Signal': signal
})

# ------------------------------------------------------------------------------
# 2. Fit Simple Linear Regression
# ------------------------------------------------------------------------------
# Fit ordinary least squares model using statsmodels formula API
linear_model = ols('Signal ~ Concentration', data=data).fit()

print("==========================================")
print("      LINEAR REGRESSION MODEL RESULTS     ")
print("==========================================")
print(linear_model.summary())

r_squared = linear_model.rsquared
print(f"\nCoefficient of Determination (R²): {r_squared:.4f}\n")

# ------------------------------------------------------------------------------
# 3. Lack-of-Fit Test (ANOVA)
# ------------------------------------------------------------------------------
# Fit full model where Concentration is treated as a categorical factor: C(Concentration)
factor_model = ols('Signal ~ C(Concentration)', data=data).fit()

# Perform Analysis of Variance comparing linear vs. factor model
lof_test = sm.stats.anova_lm(linear_model, factor_model)

print("==========================================")
print("           LACK-OF-FIT ANOVA TEST         ")
print("==========================================")
print(lof_test)

# ------------------------------------------------------------------------------
# 4. Residual Diagnostics & Plotting
# ------------------------------------------------------------------------------
# Extract fitted values and residuals
data['fitted'] = linear_model.fittedvalues
data['residuals'] = linear_model.resid

# Style setup
sns.set_theme(style="whitegrid")

# Plot 1: Calibration Curve with Linear Fit
fig1, ax1 = plt.subplots(figsize=(6, 4.5))
sns.regplot(
    data=data,
    x='Concentration',
    y='Signal',
    ax=ax1,
    color='#1F77B4',
    line_kws={'color': '#D62728'},
    scatter_kws={'alpha': 0.8, 's': 50}
)
ax1.set_title("Calibration Curve & Linearity Fit", fontweight='bold')
ax1.set_subtitle = f"R² = {r_squared:.4f}"
fig1.suptitle(f"R² = {r_squared:.4f}", fontsize=10, y=0.91, color='gray')
ax1.set_xlabel("Concentration (mg/L)")
ax1.set_ylabel("Signal Response (AU)")
plt.tight_layout()
fig1.savefig("calibration_curve.png", dpi=300)
plt.close(fig1)

# Plot 2: Residuals vs. Fitted Values
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
sns.scatterplot(
    data=data,
    x='fitted',
    y='residuals',
    ax=ax2,
    color='#1F77B4',
    s=50
)
ax2.axhline(0, linestyle='--', color='gray')
sns.regplot(
    data=data,
    x='fitted',
    y='residuals',
    ax=ax2,
    scatter=False,
    lowess=True,
    line_kws={'color': '#2CA02C', 'linestyle': ':'}
)
ax2.set_title("Residuals vs Fitted", fontweight='bold')
fig2.suptitle("Check for homoscedasticity and non-linear patterns", fontsize=10, y=0.91, color='gray')
ax2.set_xlabel("Fitted Values")
ax2.set_ylabel("Residuals")
plt.tight_layout()
fig2.savefig("residual_plot.png", dpi=300)
plt.close(fig2)

print("\nGraphics successfully saved to working directory.")
