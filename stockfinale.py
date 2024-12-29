import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Streamlit App Title
st.title("AGRAWAL'S PREDICTOR APP")

# User input for Stock Symbol
stock = st.text_input("Enter the Stock Symbol (e.g., AAPL, GOOG):", "GOOG")

# Download historical stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start=start, end=end)

# Load pre-trained model
filepath = "C:/c learinig/python/stockprediction/Latest_stock_price_model.keras"
model = load_model(filepath)

# Display the stock data
st.subheader("Historical Stock Data")
st.write(google_data)

# Calculate moving averages and visualize them
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Adj Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader("Moving Averages")
google_data['MA_250'] = google_data['Adj Close'].rolling(250).mean()
google_data['MA_200'] = google_data['Adj Close'].rolling(200).mean()
google_data['MA_100'] = google_data['Adj Close'].rolling(100).mean()

st.pyplot(plot_graph((15, 6), google_data['MA_250'], google_data))
st.pyplot(plot_graph((15, 6), google_data['MA_200'], google_data))
st.pyplot(plot_graph((15, 6), google_data['MA_100'], google_data))

# Data Preparation
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(google_data[['Adj Close']].dropna())

# Preparing the test data
splitting_len = int(len(data_scaled) * 0.7)
x_test = []
y_test = []

for i in range(100, len(data_scaled[splitting_len:])):
    x_test.append(data_scaled[splitting_len + i - 100:splitting_len + i])
    y_test.append(data_scaled[splitting_len + i])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Predictions for test data
predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

# Plot predictions vs actual values
st.subheader("Original vs Predicted Test Data")
ploting_data = pd.DataFrame(
    {
        'Original': inv_y_test.reshape(-1),
        'Predicted': inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)
st.write(ploting_data)

fig = plt.figure(figsize=(15, 6))
plt.plot(ploting_data['Original'], label='Original')
plt.plot(ploting_data['Predicted'], label='Predicted')
plt.legend()
st.pyplot(fig)

# Future Prediction: Next 30 Days
last_100_data = data_scaled[-100:].reshape(1, 100, 1)
future_prices_scaled = []

for _ in range(30):
    prediction = model.predict(last_100_data)  # Get prediction
    future_prices_scaled.append(prediction[0][0])  # Extract scalar value
    prediction_reshaped = prediction.reshape(1, 1, 1)  # Reshape for appending
    last_100_data = np.append(last_100_data[:, 1:, :], prediction_reshaped, axis=1)

# Inverse transform the scaled predictions
future_prices = scaler.inverse_transform(np.array(future_prices_scaled).reshape(-1, 1))

# Create future dates
future_dates = pd.date_range(start=google_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
future_forecast = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices.flatten()})

# Display future prices
st.subheader("Future 30-Day Price Predictions")
st.write(future_forecast)

# Plot future prices
fig = plt.figure(figsize=(15, 6))
plt.plot(future_forecast['Date'], future_forecast['Predicted Price'], label='Future Prices', color='green')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.title(f"Future Price Predictions for {stock}")
st.pyplot(fig)

# Save future predictions to CSV
future_forecast.to_csv(f"{stock}_30_day_forecast.csv", index=False)
st.write("Future prices saved to CSV file.")
