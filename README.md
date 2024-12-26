# share-market-prediction

This code is designed to predict stock prices using a Long Short-Term Memory (LSTM) model, which is a type of recurrent neural network (RNN) effective for time-series forecasting. Here's an overview of each part of the code:

1. Fetching Stock Data from Yahoo Finance
Function: fetch_stock_data(ticker_symbol, start_date="2010-01-01", end_date="2023-01-01")
Purpose: This function uses the yfinance library to fetch historical stock data (price and volume) from Yahoo Finance for a given stock ticker (e.g., "AAPL" for Apple) over a specified date range. The function returns the stock data as a pandas DataFrame.
2. Adding Technical Indicators
Function: add_technical_indicators(stock_data)
Purpose: This function adds a few commonly used technical indicators to the stock data:
50-day Simple Moving Average (SMA50): A rolling average of the closing prices over the last 50 days.
200-day Simple Moving Average (SMA200): A rolling average of the closing prices over the last 200 days.
Relative Strength Index (RSI): A momentum oscillator that measures the speed and change of price movements. It helps identify overbought or oversold conditions.
These indicators are useful in predicting stock trends and patterns.
The function returns the stock data with these added technical indicators.
3. Preprocessing the Data for LSTM
Function: preprocess_data(stock_data)
Purpose: This function prepares the data for training an LSTM model:
It selects the columns for prediction (Close, SMA50, SMA200, RSI).
Scaling: The data is scaled using the MinMaxScaler, which normalizes the values between 0 and 1 to ensure the neural network can learn efficiently.
Creating Sequences for LSTM: LSTM models take sequential data as input, so the function creates sequences of 60 time steps (60 days of data) as the input (X) and the closing price of the next day as the target (y).
Train-Test Split: The data is split into training and test datasets using train_test_split. The shuffle=False argument ensures the temporal sequence is maintained (i.e., no random shuffling of data).
It returns the preprocessed training and test data (X_train, X_test, y_train, y_test) and the scaler used for normalization.
4. Building the LSTM Model
Function: build_lstm_model(input_shape)
Purpose: This function defines the LSTM model architecture:
The model uses two LSTM layers (each with 50 units) to process the sequential data.
Dropout Layers: These are used for regularization to prevent overfitting (i.e., randomly drop some connections during training).
Dense Layer: The final output layer is a Dense layer with one unit that predicts the next closing price.
Compilation: The model is compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function (because this is a regression task).
5. Training the LSTM Model
Function: train_lstm_model(X_train, y_train)
Purpose: This function trains the LSTM model using the prepared training data (X_train and y_train). It runs for 10 epochs (iterations over the entire dataset) with a batch size of 32. After training, the model is returned.
6. Predicting Stock Prices
Function: predict_stock_price(model, X_test, y_test, scaler)
Purpose: This function makes predictions on the test data (X_test) using the trained LSTM model:
Predictions: The model predicts future stock prices (closing prices).
Inverse Scaling: The predicted and real prices are inverse transformed (converted back to the original scale) using the scaler to return the prices to their actual values.
Plotting: It then plots the predicted stock prices and the real stock prices for comparison, using matplotlib.
Main Workflow:
Fetch Stock Data:

It first fetches historical stock data for the chosen stock (e.g., Apple with ticker 'AAPL') from Yahoo Finance.
Add Technical Indicators:

It then adds the 50-day SMA, 200-day SMA, and RSI to the stock data as features that might help in making predictions.
Preprocess Data:

The stock data is preprocessed, including scaling and creating sequences of 60 days of data, where each sequence corresponds to predicting the next day's closing price.
Build and Train LSTM Model:

An LSTM model is built and trained using the preprocessed data to predict the stock price.
Prediction and Visualization:

After training, the model predicts the stock prices on the test data. The predictions are compared to the real prices, and the results are visualized with a plot showing both the predicted and actual stock prices over time.
Key Libraries Used:
yfinance: Fetches historical stock market data.
pandas: Handles data manipulation and analysis.
numpy: Handles array operations.
matplotlib: Visualizes the stock price prediction.
sklearn: Provides preprocessing functions like MinMaxScaler and train_test_split.
keras: Used to build and train the LSTM model.
