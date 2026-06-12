# Hybrid Deep Learning System for Precipitation Intensity Forecasting

24-hour precipitation intensity forecasting for 7 Ukrainian cities 
using a hybrid Transformer + XGBoost architecture.

## Overview
This system predicts hourly precipitation intensity (mm/h) for 7 Ukrainian 
cities — Kyiv, Kharkiv, Lviv, Odesa, Dnipro, Chernihiv and Uzhhorod — 
with a 24-hour forecast horizon. It combines a Transformer regression model 
for intensity prediction and an XGBoost classifier for precipitation detection.

## Architecture
- **XGBoost Classifier** — detects whether precipitation will occur
- **Transformer Regressor** — predicts hourly precipitation intensity (mm/h)
- **Hybrid Inference Pipeline** — combines both models at inference stage

## Features
- 24-hour hourly precipitation forecast
- Interactive web interface (Streamlit)
- Hourly bar charts, heat map, cumulative curve, text forecast
- CSV export of forecast data
- Forecast history log
- 7 Ukrainian cities supported

## Tech Stack
- Python, PyTorch, XGBoost, Scikit-learn
- Pandas, NumPy
- Streamlit, Plotly
- REST API (WeatherAPI)
- Docker, Git/GitHub

## Dataset
- 5 years of hourly meteorological observations (March 2021 — March 2026)
- Source: WeatherAPI
- 26 meteorological features per observation
- 7 cities across different climate zones of Ukraine

## Model Performance
| Metric | Value |
|--------|-------|
| F1-Score | 0.72 |
| ROC-AUC | 0.817 |
| PR-AUC | 0.788 |
| MAE improvement vs baseline | 41% |

## Installation
```bash
git clone https://github.com/Nahaivska13/Hybrid-Deep-Learning-System-for-Precipitation-Intensity-Forecasting.git
cd Hybrid-Deep-Learning-System-for-Precipitation-Intensity-Forecasting
conda create -n precip python=3.12
conda activate precip
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```
Then in the sidebar:
1. Click **Download new data**
2. Click **Process data**
3. Click **Scaling**
4. Click **Update forecast**

