# Automated DCF & Sensitivity Engine

## Overview
A production-grade **Discounted Cash Flow (DCF)** framework that automates equity valuation and performs **algorithmic sensitivity analysis** to estimate intrinsic value.

### Key Features
- **Live Pipeline:** Automated data extraction (Income, Balance, Cash Flow) via `yfinance`.
- **Dynamic WACC:** Real-time CAPM calculation (Beta, Rf, ERP).
- **Backtesting:** 70/30 split logic to validate FCF growth projections.
- **Optimization:** **Optuna** grid sweep (100 trials) across WACC and Terminal Growth.
- **Analytics:** Automated dashboard (Heatmap, Distribution, and Scorecard).

## Architecture
- `DCFConfig`: Centralized hyperparameters.
- `DCFEngine`: Mathematical core (UFCF, WACC, Gordon Growth).
- `SensitivityOptimizer`: Optuna-based parameter sweep.
- `Visualizer`: High-fidelity financial reporting.

## AAPL Case Study Results
- **Intrinsic Value:** $197.31 (Market: $204.50)
- **Margin of Safety:** -3.5% (Fairly Valued)
- **Valuation Range:** $141 (Bear) to $282 (Bull)

## Quick Start
1. `pip install -r requirements.txt`
2. `python main.py --ticker AAPL`

---
**Author:** TUNG Alexandre | **Affiliation:** EFREI Paris
