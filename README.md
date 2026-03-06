# Satellite Parking Lot Momentum (SPLM) 

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/AI-YOLOv8-green?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Quant Finance](https://img.shields.io/badge/Quant-Backtesting-orange?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
**SPLM** is a quantitative research framework designed to generate stock market alpha using alternative data. The project builds an end-to-end pipeline that counts vehicles in satellite imagery of retail parking lots to predict "Physical Activity" (PAI). By comparing this real-world traffic data against institutional analyst sentiment (AMI), the system identifies **Alpha Divergences**—predicting stock price movements before they are reflected in analyst reports.

This project demonstrates expertise in **Computer Vision**, **Modular Software Engineering**, and **Quantitative Portfolio Construction**.

---

## Core Features
- **Computer Vision Pipeline**: Automated car counting using **YOLOv8** on multi-spectral satellite imagery patches.
- **Alternative Data Signal Construction**: Weather-normalized, Z-scored signal processing to build the **Physical Activity Index (PAI)**.
- **Sentiment Divergence Engine**: Comparison of real-world foot traffic vs. **IBES Analyst Estimates** to find "leading" market indicators.
- **Quant Backtest Engine**: Vectorized backtester with realistic transaction cost modeling (BPS-based) and turnover analysis.
- **Risk Management**: Sector-neutral and dollar-neutral long/short portfolio optimization using `cvxpy`.

---

## Tech Stack & Architecture

### Languages & Tools
- **Language**: Python 3.12
- **Data Engineering**: Pandas, NumPy, Scipy
- **Computer Vision**: Ultralytics (YOLOv8)
- **Finance Engine**: Statsmodels, CVXPY (Optimization), Matplotlib/Seaborn (Visualization)
- **GIS**: OpenStreetMap Overpass API (Parking Lot Geometry Retrieval)

### Project Structure
```text
satellite_parking_momentum/
├── data/               # Raw and processed datasets
├── notebooks/          # Research Flow (Signal Validation & Backtesting)
├── pipeline/           # Orchestration scripts (Daily runner)
├── backtest/           # Core backtesting and cost modeling logic
├── signals/            # Alpha signal construction (PAI, AMI)
├── cv_models/          # YOLOv8 inference and spectral filtering
└── requirements.txt    # Project dependencies
```

---

## Installation & Setup (Hybrid MSYS2/Windows)

This project is optimized for Windows power users using **MSYS2/MinGW**. It uses a hybrid environment approach to handle heavy scientific dependencies efficiently.

### 1. Install Heavy Dependencies (MSYS2 Terminal)
Run the following in your MSYS2 MinGW64 terminal to avoid SSL/Build issues:
```bash
pacman -S mingw-w64-x86_64-python-pandas mingw-w64-x86_64-python-numpy \
          mingw-w64-x86_64-python-statsmodels mingw-w64-x86_64-python-scipy \
          mingw-w64-x86_64-python-ipykernel
```

### 2. Configure Virtual Environment (PowerShell)
```powershell
python -m venv venv --system-site-packages
.\venv\bin\Activate.ps1
pip install -r requirements.txt --prefer-binary
```

---

## Running the Pipeline

### Daily Signal Generation
To run the automated research flow that processes imagery and generates trade targets:
```powershell
python pipeline/daily_runner.py
```

### Visualizing Results
Explore the saved results in the **Jupyter Notebooks**:
1.  **[Signal Alpha Analysis](notebooks/01_signal_validation.ipynb)**: Check Information Coefficient (IC) decay and signal predictive power.
2.  **[Backtest Results](notebooks/02_backtest_analysis.ipynb)**: View the Equity Curve, Drawdowns, and Net Returns after transaction costs.

---

## Key Learnings & Future Scope
- **Alternative Data**: Faced and solved "look-ahead bias" and "point-in-time" data constraints in finance.
- **Software Design**: Implemented a **Strict Modular Architecture** allowing easy swapping of CV models or data providers.
- **Future Improvements**: 
  - Integrating real-time Planet Labs API hooks.
  - Upgrading to a distributed worker system (Celery/RabbitMQ) for large-scale GIS processing.

---

