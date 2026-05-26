import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import time
from datetime import datetime
import pytz

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Stock Market Dashboard",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("📈 AI Stock Market Prediction Dashboard")

st.write(
    "Semi-Live AIML Market Prediction System"
)

# =========================
# AUTO REFRESH
# =========================

st.caption("Dashboard auto-refreshes every 60 seconds")

time.sleep(1)

# =========================
# LOAD LIVE DATA
# =========================

ticker = "AAPL"

try:

    data = yf.download(
        ticker,
        period="5d",
        interval="5m",
        progress=False
    )

    # RESET INDEX

    data.reset_index(inplace=True)

    # FIX MULTI INDEX

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

except Exception as e:

    st.error(f"Error downloading stock data: {e}")
    st.stop()

# CHECK TIME COLUMN

possible_time_columns = [
    "Datetime",
    "Date",
    "index"
]

time_column = None

for col in possible_time_columns:

    if col in data.columns:
        time_column = col
        break

# IF STILL NONE

if time_column is None:

    st.write("Available Columns:")
    st.write(data.columns)

    st.error(
        "No valid datetime column found."
    )

    st.stop()

# =========================
# TIMEZONE CONVERSION
# =========================

try:

    data[time_column] = pd.to_datetime(
        data[time_column],
        utc=True
    )

    data[time_column] = (
        data[time_column]
        .dt.tz_convert("Asia/Kolkata")
    )

except Exception as e:

    st.warning(f"Timezone conversion skipped: {e}")

# =========================
# FEATURE ENGINEERING
# =========================

try:

    # RETURNS

    data["Returns"] = (
        data["Close"].pct_change()
    )

    # MOVING AVERAGES

    data["MA_5"] = (
        data["Close"].rolling(5).mean()
    )

    data["MA_20"] = (
        data["Close"].rolling(20).mean()
    )

    # VOLATILITY

    data["Volatility"] = (
        data["Returns"].rolling(10).std()
    )

    # MOMENTUM

    data["Momentum"] = (
        data["Close"] - data["Close"].shift(5)
    )

    # VOLUME CHANGE

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    # RSI

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # MACD

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    # REMOVE NaN VALUES

    data.dropna(inplace=True)

except Exception as e:

    st.error(f"Feature engineering failed: {e}")
    st.stop()

# =========================
# LOAD MODELS
# =========================

try:

    xgb_model = joblib.load(
        "models/xgb_model.pkl"
    )

except Exception as e:

    st.error(f"XGBoost model loading failed: {e}")
    st.stop()

# OPTIONAL REGRESSION MODEL

reg_model_exists = True

try:

    reg_model = joblib.load(
        "models/regression_model.pkl"
    )

except:

    reg_model_exists = False

# =========================
# SHOW LIVE DATA
# =========================

st.subheader("📊 Latest Live Market Data")

st.dataframe(
    data.tail(10)
)

# =========================
# LIVE STOCK CHART
# =========================

st.subheader("📉 Live Stock Price")

fig, ax = plt.subplots(figsize=(10,4))

ax.plot(
    data[time_column],
    data["Close"]
)

ax.set_title("AAPL Live Close Price")

ax.set_xlabel("Time (IST)")

ax.set_ylabel("Price")

plt.xticks(rotation=45)

st.pyplot(fig)

# =========================
# FEATURE SELECTION
# =========================

feature_columns = [
    "Returns",
    "MA_5",
    "MA_20",
    "Volatility",
    "RSI",
    "MACD",
    "Momentum",
    "Volume_Change"
]

# =========================
# LATEST DATA FOR PREDICTION
# =========================

latest_data = data.tail(1)

X_latest = latest_data[
    feature_columns
]

# =========================
# AI PREDICTION
# =========================

prediction = xgb_model.predict(
    X_latest
)

prediction_map = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP"
}

final_prediction = prediction_map[
    prediction[0]
]

# =========================
# CONFIDENCE SCORE
# =========================

probabilities = xgb_model.predict_proba(
    X_latest
)

confidence_score = round(
    np.max(probabilities) * 100,
    2
)

# =========================
# EXPECTED RETURN
# =========================

if reg_model_exists:

    predicted_return = reg_model.predict(
        X_latest
    )

    expected_return = round(
        predicted_return[0],
        2
    )

else:

    expected_return = "N/A"

# =========================
# MARKET REGIME
# =========================

volatility = (
    data["Close"]
    .pct_change()
    .std()
)

if volatility > 0.01:

    market_regime = "High Volatility"

else:

    market_regime = "Calm Market"

# =========================
# FINAL OUTPUT
# =========================

st.subheader("🎯 Final AI Prediction")

final_output = pd.DataFrame({

    "Prediction": [final_prediction],

    "Confidence": [
        f"{confidence_score}%"
    ],

    "Expected Return": [
        f"{expected_return}%"
    ],

    "Market Regime": [
        market_regime
    ]

})

st.table(final_output)

# =========================
# CONFIDENCE CHART
# =========================

st.subheader("📌 Prediction Confidence")

fig2, ax2 = plt.subplots(figsize=(5,3))

ax2.bar(
    ["Confidence"],
    [confidence_score]
)

ax2.set_ylabel("Percentage")

ax2.set_title("Model Confidence")

st.pyplot(fig2)

# =========================
# MARKET REGIME SECTION
# =========================

st.subheader("🌍 Market Regime")

st.info(
    f"Current market regime: {market_regime}"
)

# =========================
# LAST UPDATE TIME
# =========================

st.subheader("🕒 Latest Update")

st.success(
    f"Latest Market Timestamp (IST): {data[time_column].iloc[-1]}"
)

# =========================
# CURRENT IST TIME
# =========================

st.subheader("🕒 Current IST Time")

india = pytz.timezone(
    "Asia/Kolkata"
)

current_time = datetime.now(
    india
)

st.success(
    f"Current IST Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
)

# =========================
# MARKET DATA TIMESTAMP
# =========================

st.subheader("📌 Latest Market Candle")

st.info(
    f"Latest Market Timestamp: {data[time_column].iloc[-1]}"
)

# =========================
# FINAL CONCLUSION
# =========================

st.subheader("✅ AI Conclusion")

st.success(
    f"""
    The AI system predicts a {final_prediction}
    movement with {confidence_score}% confidence
    under {market_regime} conditions.
    """
)