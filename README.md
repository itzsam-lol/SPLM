# SPLM India: Satellite Parking Lot Momentum for Emerging Markets

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-Active-4285F4?style=flat-square&logo=googleearth)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research_Ready-brightgreen?style=flat-square)

## Overview
**SPLM India** is a quantitative research framework designed to predict revenue surprises in Indian organized retail stocks using alternative data. By leveraging open-source 10-meter satellite imagery (Sentinel-2) and spectral classification techniques (NDBI), the pipeline measures physical activity (parking lot occupancy) at major flagship retail locations and translates these observations into a tradable alpha signal.

This repository adapts traditional high-resolution satellite momentum strategies (typically reliant on costly commercial imagery like Planet Labs or Maxar) into a fully accessible, free-to-operate model tailored for emerging markets. 

## Key Features
- **Spectral Computer Vision Pipeline**: Automatically processes 10m Sentinel-2 multi-spectral imagery to compute Normalized Difference Built-Up Index (NDBI) as a proxy for vehicle density.
- **Automated Earth Engine Orchestration**: Natively interacts with Google Earth Engine (GEE) to fetch, process, and chip satellite composites precisely aligned with OpenStreetMap (OSM) retail polygons.
- **Robust NSE Data Architecture**: Ingests actual revenue figures from the National Stock Exchange (NSE) APIs with fail-safes (yfinance integrations and structural quantitative proxy targets) to ensure resilient backtesting.
- **End-to-End Automation**: The `quarterly_runner.py` orchestrates the complete physical-to-financial divergence workflow across an expandable universe of 12 top Indian retail equities (e.g., DMART, TRENT, JUBLFOOD).

## Pipeline Architecture
1. **Universe & Location Registry**: Defines the sector-mapped universe of Indian retail stocks and utilizes OSM Overpass to map physical parking geometries.
2. **Sentinel Downloader**: Uses GEE to fetch non-cloudy, quarterly harmonized composites of the target geolocations.
3. **Occupancy Extractor**: Calculates average spectral "hardness" (occupancy proxy) within the bounded retail polygons.
4. **Signal Constructors**: 
   - *Physical Activity Index (PAI)*: Normalizes trailing multi-quarter occupancy data against seasonal offsets.
   - *Analyst Momentum Index (AMI)*: Derives base analyst sentiment and target revenue trajectories.
5. **Divergence Engine**: Aggregates the PAI and AMI to calculate the cross-sectional momentum differential, outputting normalized directional signals (-1 to +1).

## Installation & Setup

### Prerequisites
- Python 3.12+
- A Google Cloud Project with the **Earth Engine API** enabled.

### Environment
1. Clone the repository and initialize a virtual environment:
```bash
git clone https://github.com/itzsam-lol/SPLM.git
cd satellite_parking_momentum
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install strictly versioned dependencies (critical for pipeline stability):
```bash
pip install -r requirements.txt
```

3. Authenticate Earth Engine (Required on first run):
```bash
earthengine authenticate
```

## Usage
To execute the fully automated pipeline for the entire tracking universe over a specific date range:

```bash
python pipeline/quarterly_runner.py --start-year 2022 --end-year 2023
```

The pipeline will log extraction rates and output the final signal vectors to:
`data/processed/final_signals.csv`

To test the computer vision layer specifically against a single ticker:
```bash
python models/occupancy_cv.py --ticker DMART --year 2023 --quarter 1
```

## Research Outputs
The pipeline generates a comprehensive set of research artifacts in the `results/` directory, structured for direct integration into academic publications (e.g., ICAIF):

- **`results/00_validation/`**: Contains PIT (Point-in-Time) validation logs and test suite outputs ensuring zero lookahead bias.
- **`results/02_tables/`**: High-fidelity CSV tables for Data Coverage, Case Study Observations, and IC Analysis.
- **`results/03_figures/`**: Publication-ready visualizations including Pipeline Architecture, PAI Time Series, and IC Decay curves.
- **`paper_results_summary.json`**: A consolidated JSON artifact containing all key performance metrics and coefficients.

To regenerate the full suite of research results, execute:
```bash
python generate_results.py
```

For a detailed breakdown of the model's accuracy, constraints (such as Indian Monsoon occlusion windows), and initial predictive metrics, refer to the included **SPLM India Research Report (PDF)**.

## Ethics & Disclaimer
This framework was engineered strictly for academic research and methodological demonstration. The synthetic fallback layers and proxy estimators are designed to test the architectural soundness of the divergence model when proprietary earnings or analyst data is unavailable. It does not constitute financial advice.
