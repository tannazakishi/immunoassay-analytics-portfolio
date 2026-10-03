
"""
Title: 4PL Non-Linear Regression Pipeline for Immunoassay Analysis
Description: Fits dose-response curves using a 4-Parameter Logistic (4PL) model 
             for ELISA and chemiluminescent diagnostic platforms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistic_4pl(x, A, B, C, D):
    """
    4-Parameter Logistic Equation:
    A = Minimum asymptote (background)
    B = Hill slope
    C = Inflection point (EC50 / IC50)
    D = Maximum asymptote (saturation)
    """
    return A + (D - A) / (1.0 + np.exp(B * (np.log(x) - np.log(C))))

def analyze_assay_curve(concentrations, signals):
    """
    Fits 4PL model to assay concentration and signal data.
    """
    # Initial parameter guesses [A, B, C, D]
    p0 = [min(signals), 1.0, np.median(concentrations), max(signals)]
    
    # Perform curve fitting
    popt, pcov = curve_fit(logistic_4pl, concentrations, signals, p0=p0, maxfev=10000)
    
    print("--- Fitted 4PL Parameters ---")
    print(f"Min Asymptote (A): {popt[0]:.4f}")
    print(f"Hill Slope (B):    {popt[1]:.4f}")
    print(f"Inflection (C):    {popt[2]:.4f}")
    print(f"Max Asymptote (D): {popt[3]:.4f}")
    
    return popt

# Example execution with dummy standard curve data
if __name__ == "__main__":
    # Standard concentrations (ng/mL)
    concs = np.array([0.1, 0.39, 1.56, 6.25, 25.0, 100.0])
    # Corresponding chemiluminescent signals (RLU)
    sigs = np.array([120, 450, 1850, 7200, 24500, 48000])
    
    parameters = analyze_assay_curve(concs, sigs)

