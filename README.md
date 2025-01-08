# Cross-Impact Analysis of Order Flow Imbalance (OFI)

This project analyzes the cross-impact of **Order Flow Imbalance (OFI)** on short-term price changes across multiple stocks. The analysis includes:
- Computation of multi-level OFI metrics and integration using **Principal Component Analysis (PCA)**.
- **Contemporaneous** and **lagged cross-impact** analysis.
- **Sector-level** cross-impact analysis.
- **Feature importance** analysis using Random Forest models.

The project is implemented in Python and uses high-frequency equity market data from the Nasdaq 100.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Running the Analysis](#running-the-analysis)
5. [Folder Structure](#folder-structure)
6. [Results](#results)
7. [Key Findings](#key-findings)
8. [References](#references)
9. [Running on Google Cloud VM](#running-on-google-cloud-vm)

---

## Project Overview

The goal of this project is to analyze how **Order Flow Imbalance (OFI)** affects short-term price changes across multiple stocks. The analysis is based on the methodologies outlined in the paper *"Cross-Impact of Order Flow Imbalance in Equity Markets"*. Key tasks include:
1. **Computing OFI Metrics:** Derive multi-level OFI metrics (up to 5 levels) for each stock and integrate them into a single metric using PCA.
2. **Analyzing Cross-Impact:** Examine the contemporaneous and lagged cross-impact of OFI on price changes across stocks.
3. **Sector-Level Analysis:** Explore how cross-impact varies across different sectors (e.g., Tech, Healthcare, Energy).
4. **Feature Importance Analysis:** Evaluate the importance of OFI, volume, and volatility in predicting price changes using Random Forest models.

---

## Dataset

The dataset used in this project consists of high-frequency equity market data for five highly liquid stocks from the Nasdaq 100, representing various sectors:
- **Tech:** AAPL
- **Healthcare:** AMGN
- **Consumer Discretionary:** TSLA
- **Financials:** JPM
- **Energy:** XOM

The data includes:
- **Order book updates** (up to 5 levels of depth).
- **Trades** over a time period of 1 week or 1 month.

The data can be obtained from [Databento](https://databento.com/) using the Nasdaq TotalView-ITCH catalog and the MBP-10 schema.

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cross-impact-analysis.git
cd cross-impact-analysis
```
### 2. Install Dependencies
Install the required Python packages that are essential for running the code:
```bash
pip install -r requirements.txt
```
## Running the Analysis
### 1. Preprocess the Data
Run the data preprocessing script to clean and prepare the data:
```bash
python scripts/data_preprocessing.py
```
This will generate cleaned data files in the data/cleaned folder.
### 2. Compute OFI Metrics
Compute multi-level OFI metrics and integrate them using PCA:
```bash
python scripts/compute_ofi.py
```
This will generate processed data files in the data/processed folder.
### Run Cross-Impact Analysis
- Contemporaneuous Cross-Impact:
```bash
python scripts/contemp_cross_impact.py
```
- Lagged Cross-Impact:
```bash
python scripts/cross_impact_linear.py
```
- Lagged Random Forest Model:
```bash
python scripts/cross_impact_forest.py
```
### 4. Analyze Sector-Level Cross-Impact
```bash
python scripts/sector_cross_impact.py
```
### 5. Perform Feature Importance Analysis
```bash
python scripts/feature_importance.py
```
## Results
The results of the analysis are saved in the results/ filder, organized into subfolders:
- Contemporaneous Cross-Impact: Regression results, visualizations, and summary statistics.
- Lagged Cross-Impact: Regression results, R-squared plots, and best lag analysis.
- Sector-Level Analysis: Heatmaps and summary statistics for sector-level cross-impact.
- Feature Importance: Bar plots and tables showing the importance of OFI, volume, and volatility.

project/
├── data/                   
│   ├── raw/                # Raw data files
│   ├── cleaned/            # Cleaned data files
│   ├── processed/          # Processed data files (OFI metrics, PCA)
|   ├── parquet/            # Processed data converted to parquet
├── scripts/                # Python scripts for modular implementations
│   ├── compute_ofi.py      # Compute OFI metrics and integrate using PCA
│   ├── data_preprocessing.py # Clean and preprocess raw data
│   ├── contemp_cross_impact.py # Contemporaneous cross-impact analysis
│   ├── cross_impact_linear.py # Lagged cross-impact analysis using linear regression
│   ├── cross_impact_forest.py # Lagged cross-impact analysis using Random Forest
│   ├── sector_cross_impact.py # Sector-level cross-impact analysis
│   ├── feature_importance.py # Feature importance analysis using Random Forest
├── results/                # Outputs (e.g., figures, tables, analysis results)
│   ├── contemporaneous_cross_impact/
│   ├── lagged_linear/
│   ├── sector_analysis/
│   ├── feature_importance/
├── README.md               # Detailed instructions on how to run the code
├── requirements.txt        # List of Python packages used in the project
