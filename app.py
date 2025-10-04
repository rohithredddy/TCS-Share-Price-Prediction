# app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from snowflake_connector import write_predictions_to_snowflake

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Prediction", layout="centered")
st.title("📈 Stock Market Price Prediction using LSTM")
st.markdown("""
This web app predicts the next N-day closing prices of a stock using a trained LSTM model.
**Note:** the provided `keras_model.h5` was trained on TCS.NS data. Predictions for other tickers are experimental unless the model is retrained on them.
""")

# Input
ticker = st.text_input("Enter Stock Symbol (e.g., TCS.NS, AAPL, INFY.NS):", "TCS.NS")
days = st.slider("Prediction Horizon (days):", 1, 10, 5)

# Cached model loader to avoid reloading repeatedly
@st.cache_resource(show_spinner=False)
def load_trained_model(path="keras_model.h5"):
    return load_model(path)

# Get snowflake creds from st.secrets (preferred on Streamlit Cloud) or env vars
def get_snowflake_creds():
    # If deployed in Streamlit Cloud, use st.secrets:
    if "snowflake" in st.secrets:
        return st.secrets["snowflake"]
    # fallback to environment variables
    return {
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
        "role": os.environ.get("SNOWFLAKE_ROLE")
    }

if st.button("🔮 Predict"):
    st.write(f"Fetching latest data for **{ticker}** ...")

    # Load model (cached)
    try:
        model = load_trained_model("keras_model.h5")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    # Fetch data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=200)
    df = yf.download(ticker, start=start_date, end=end_date)

    if df is None or df.empty:
        st.error("No data found for the given ticker. Please check the symbol and try again.")
        st.stop()

    close_data = df[['Close']].dropna()

    # Ensure we have enough history for the time_step
    time_step = 100
    if len(close_data) < time_step:
        st.error(f"Not enough historical data for prediction (need at least {time_step} rows).")
        st.stop()

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    # Predict recursively
    last_seq = scaled_data[-time_step:].copy()
    future_preds = []
    try:
        for _ in range(days):
            X_input = np.array([last_seq])
            next_pred = model.predict(X_input)[0]
            future_preds.append(next_pred)
            last_seq = np.concatenate((last_seq[1:], next_pred.reshape(1, 1)), axis=0)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Inverse transform predictions
    future_prices = scaler.inverse_transform(future_preds)
    future_dates = [end_date + timedelta(days=i+1) for i in range(days)]
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices.flatten()})

    # Show results
    st.subheader("📅 Predicted Prices:")
    st.dataframe(future_df)

    # Plot results (historical + forecast)
    # --- Safe Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

# Plot last 60 actual closing prices
    ax.plot(close_data['Close'].tail(60).values, label="Last 60 Days (Actual)", linewidth=2)

# Join last actual + predicted points
    last_close = float(close_data['Close'].iloc[-1])
    pred_series = np.concatenate(([last_close], future_prices.flatten()))

    # Create X positions that continue from the last actual point
    x_future = np.arange(len(close_data['Close'].tail(60)) - 1, len(close_data['Close'].tail(60)) - 1 + len(pred_series))

    ax.plot(x_future, pred_series, 'r--o', label="Predicted (Next Days)", linewidth=2)

    # Formatting
    ax.set_title(f"{ticker} — Actual vs Predicted")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)


    # Write to Snowflake
    with st.spinner("Saving predictions to Snowflake..."):
        creds = get_snowflake_creds()
        success = write_predictions_to_snowflake(future_df, table_name='TCS_PREDICTIONS', creds=creds)
        if success:
            st.success("✅ Predictions saved to Snowflake Cloud Database!")
        else:
            st.warning("⚠️ Could not save predictions. Check Snowflake connection and logs.")
