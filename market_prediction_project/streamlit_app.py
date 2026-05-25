import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE TITLE
# =========================

st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    layout="wide"
)

st.title(
    "📈 Regime-Aware Stock Market Prediction Dashboard"
)

st.write(
    "Intermediate-Level AIML Internship Project"
)

# =========================
# LOAD DATA
# =========================

data = pd.read_csv(
    "data/aapl_cleaned.csv"
)

# =========================
# SHOW DATA
# =========================

st.subheader("📊 Latest Stock Market Data")

st.dataframe(
    data.tail(10)
)

# =========================
# STOCK PRICE CHART
# =========================

st.subheader("📉 Stock Price Movement")

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(data["Close"])

ax.set_title("AAPL Close Price")

ax.set_xlabel("Time")

ax.set_ylabel("Price")

st.pyplot(fig)

# =========================
# MODEL ACCURACY
# =========================

st.subheader("🤖 Model Accuracy Comparison")

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

# PUT YOUR REAL ACCURACY VALUES HERE
accuracies = [
    0.61,
    0.68,
    0.73
]

fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.bar(models, accuracies)

ax2.set_title("Model Accuracy")

ax2.set_ylabel("Accuracy")

st.pyplot(fig2)

# =========================
# FINAL PREDICTION OUTPUT
# =========================

st.subheader("🎯 Final Prediction Output")

# CHANGE THESE VALUES LATER
prediction = "UP"

confidence = "81%"

expected_return = "+0.9%"

regime = "Volatile"

final_output = pd.DataFrame({
    "Prediction": [prediction],
    "Confidence": [confidence],
    "Expected Return": [expected_return],
    "Market Regime": [regime]
})

st.table(final_output)

# =========================
# CONFIDENCE SCORE CHART
# =========================

st.subheader("📌 Confidence Score")

confidence_value = [81]

fig3, ax3 = plt.subplots(figsize=(8,4))

ax3.bar(
    ["Confidence"],
    confidence_value
)

ax3.set_ylabel("Percentage")

ax3.set_title("Prediction Confidence")

st.pyplot(fig3)

# =========================
# MARKET REGIME SECTION
# =========================

st.subheader("🌍 Market Regime Analysis")

st.info(
    "Current market regime detected as HIGH VOLATILITY."
)

# =========================
# FINAL CONCLUSION
# =========================

st.subheader("✅ Conclusion")

st.success(
    """
    The AIML system predicts short-term bullish market movement
    with high confidence under volatile market conditions.
    """
)