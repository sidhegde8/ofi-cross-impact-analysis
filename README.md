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
