import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers

# 1. Fetch Stock Data from Yahoo Finance
def fetch_stock_data(ticker_symbol, start_date="2010-01-01", end_date="2023-01-01"):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data

# 2. Add Technical Indicators (e.g., Moving Averages, RSI)
def add_technical_indicators(stock_data):
    # Calculate the 50-day and 200-day Simple Moving Averages
    stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()

    # Calculate Relative Strength Index (RSI)
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    return stock_data

# 3. Preprocess Data for LSTM Model
def preprocess_data(stock_data):
    # Use Close price for prediction
    data = stock_data[['Close', 'SMA50', 'SMA200', 'RSI']].dropna()

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create a dataset for LSTM (X is the input and Y is the future price)
    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])  # 60 days of data as input
        y.append(scaled_data[i, 0])    # Predict the closing price

    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

# 4. Build and Train LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()

    # LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout for regularization

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Fully connected layer
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# 5. Train the LSTM model
def train_lstm_model(X_train, y_train):
    model = build_lstm_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

# 6. Predict Future Stock Prices
def predict_stock_price(model, X_test, y_test, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(np.column_stack((predicted_prices, np.zeros((predicted_prices.shape[0], 2)))))
    
    real_prices = scaler.inverse_transform(np.column_stack((y_test, np.zeros((y_test.shape[0], 2)))))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices[:, 0], color='blue', label='Real Stock Price')
    plt.plot(predicted_prices[:, 0], color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main function to execute the workflow
if __name__ == "__main__":
    # Choose your stock ticker (e.g., 'AAPL' for Apple)
    ticker_symbol = 'AAPL'
    
    # Step 1: Fetch stock data
    stock_data = fetch_stock_data(ticker_symbol, start_date="2010-01-01", end_date="2023-01-01")
    
    # Step 2: Add technical indicators (SMA, RSI)
    stock_data = add_technical_indicators(stock_data)

    # Step 3: Preprocess the data for LSTM
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)

    # Step 4: Train LSTM model
    model = train_lstm_model(X_train, y_train)

    # Step 5: Make predictions
    predict_stock_price(model, X_test, y_test, scaler)
