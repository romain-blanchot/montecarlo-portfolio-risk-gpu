"""Simulation parameters for the 8-asset benchmark portfolio.

Edit the values here, then run:

    python scripts/demo.py
"""

import numpy as np

# --- Asset universe (8 assets across 5 sectors) ---
ASSET_NAMES = ["AAPL", "MSFT", "JPM", "GS", "XOM", "JNJ", "GLD", "BND"]

# Approximate current prices in USD
S0 = np.array([195.0, 415.0, 205.0, 510.0, 118.0, 155.0, 215.0, 73.0])

# Allocation weights — must sum to 1.0
WEIGHTS = np.array([0.18, 0.17, 0.12, 0.10, 0.10, 0.10, 0.13, 0.10])

# --- Annualised market parameters (based on 2015–2024 historical estimates) ---

MU = np.array(
    [
        0.22,  # AAPL
        0.25,  # MSFT
        0.12,  # JPM
        0.10,  # GS
        0.07,  # XOM
        0.08,  # JNJ
        0.05,  # GLD
        0.03,  # BND
    ]
)

SIGMA = np.array(
    [
        0.28,  # AAPL
        0.26,  # MSFT
        0.24,  # JPM
        0.27,  # GS
        0.26,  # XOM
        0.15,  # JNJ  — defensive, low vol
        0.15,  # GLD
        0.06,  # BND  — very low vol
    ]
)

# Simulation horizon: 1 trading year (252 daily steps)
N_STEPS = 252
DT = 1.0 / 252

# --- Normal-market correlation matrix (8×8) ---
# Tech names are highly correlated; gold and bonds diverge from equities
CORR = np.array(
    [
        #  AAPL   MSFT   JPM    GS     XOM    JNJ    GLD    BND
        [1.00, 0.78, 0.48, 0.42, 0.22, 0.28, -0.05, -0.22],  # AAPL
        [0.78, 1.00, 0.45, 0.40, 0.18, 0.30, -0.08, -0.25],  # MSFT
        [0.48, 0.45, 1.00, 0.72, 0.35, 0.32, 0.05, -0.15],  # JPM
        [0.42, 0.40, 0.72, 1.00, 0.30, 0.28, 0.08, -0.18],  # GS
        [0.22, 0.18, 0.35, 0.30, 1.00, 0.25, 0.35, -0.10],  # XOM
        [0.28, 0.30, 0.32, 0.28, 0.25, 1.00, 0.15, 0.10],  # JNJ
        [-0.05, -0.08, 0.05, 0.08, 0.35, 0.15, 1.00, 0.35],  # GLD
        [-0.22, -0.25, -0.15, -0.18, -0.10, 0.10, 0.35, 1.00],  # BND
    ]
)

# --- Stress scenario: systemic financial crisis ---
# All equity correlations spike (flight to quality effect);
# gold and bonds decouple even further from equities.
CORR_STRESS = np.array(
    [
        #  AAPL   MSFT   JPM    GS     XOM    JNJ    GLD    BND
        [1.00, 0.95, 0.82, 0.80, 0.70, 0.65, -0.30, -0.55],  # AAPL
        [0.95, 1.00, 0.80, 0.78, 0.68, 0.63, -0.32, -0.57],  # MSFT
        [0.82, 0.80, 1.00, 0.92, 0.72, 0.68, -0.10, -0.40],  # JPM
        [0.80, 0.78, 0.92, 1.00, 0.70, 0.65, -0.08, -0.42],  # GS
        [0.70, 0.68, 0.72, 0.70, 1.00, 0.60, 0.20, -0.30],  # XOM
        [0.65, 0.63, 0.68, 0.65, 0.60, 1.00, 0.05, -0.10],  # JNJ
        [-0.30, -0.32, -0.10, -0.08, 0.20, 0.05, 1.00, 0.55],  # GLD
        [-0.55, -0.57, -0.40, -0.42, -0.30, -0.10, 0.55, 1.00],  # BND
    ]
)

# --- Simulation settings ---
N_PATHS = 1_000_000
SEED = 42

CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Number of paths used for the CPU vs GPU benchmark
BENCHMARK_PATHS = 200_000
