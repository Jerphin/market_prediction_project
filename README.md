# AI Stock Market Prediction Dashboard

An intermediate-level AIML project that predicts short-term stock market movement using Machine Learning, live market data, feature engineering, regime detection, and a professional Streamlit dashboard.

---

# Project Overview

This project is a semi-live AI-powered stock market prediction system built using:

- Python
- Machine Learning
- XGBoost
- Streamlit
- Yahoo Finance API

The system fetches live stock market data, performs feature engineering, predicts market movement, calculates confidence scores, detects market regimes, and visualizes everything inside a professional dashboard.

---

# Features

✅ Semi-live stock market data using Yahoo Finance  
✅ Feature engineering for market indicators  
✅ XGBoost prediction model  
✅ Confidence score using predict_proba()  
✅ Market regime detection  
✅ Expected return prediction  
✅ Professional Streamlit dashboard  
✅ Real-time visualization  
✅ Model saving/loading using Joblib  

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Streamlit
- yFinance
- Joblib

---

# Machine Learning Workflow

Live Market Data  
↓  
Feature Engineering  
↓  
Model Training  
↓  
Prediction Generation  
↓  
Confidence Scoring  
↓  
Regime Detection  
↓  
Dashboard Visualization  

---

# Feature Engineering

The project generates important technical indicators such as:

- Returns
- Moving Average (MA_5)
- Moving Average (MA_20)
- RSI
- MACD
- Volatility
- Momentum
- Volume Change

These features help the ML model understand market behavior more effectively.

---

# Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline classification |
| Random Forest | Ensemble learning |
| XGBoost | Final prediction model |
| Regression Model | Expected return prediction |

---

# Dashboard Outputs

The dashboard displays:

- Latest live stock data
- Stock price chart
- AI prediction (UP / DOWN / NEUTRAL)
- Confidence score
- Expected return prediction
- Market regime
- Current IST time
- Latest market candle timestamp

---

# Market Regime Detection

The project detects:

- Calm Market
- High Volatility Market

using rolling volatility calculations.

---

# Confidence Score

The confidence score is generated using:

```python
predict_proba()
