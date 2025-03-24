"""
Tesla Stock Trading Agent with Machine Learning

This script simulates a trading agent that predicts Tesla (TSLA) stock prices
and decides daily trading actions (Buy, Sell, Hold) over a set simulation period.

Key Features:
- Uses yahoo finance to pull live TESLA stock data
- Computes RSI and 5-day SMA as technical indicators and features for our agent
- Trains a Random Forest Regressor Model to predict next-day stock prices
- Makes trading decisions based on predicted movement
- Tracks daily trades, calculates final profit/loss for performance metric indication
- Visualizes buy/sell points on a chart as well, will show a line on the graph for 5 day predictions and a single point on the graph for each daily 9 am prediction

Machine Learning and Data Mining
Group #8: Ashad Ahmed (100745913), Andy Dai(100726784)

"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Simulation Configuration
# Set the starting constraints (starting balance and transaction fee) and the time window for simulated trading
STARTING_BALANCE = 10000
TRANSACTION_FEE = 0.01  # 1%
TRADING_DAYS = pd.date_range(start="2025-03-10", end="2025-03-10", freq='B')  # Changing these dates each day before 9 AM during testing week

# Initialize Portfolio State
# Tracks balance, shares held, and daily trade logs
balance = STARTING_BALANCE
stocks_owned = 0
trade_log = []

# Obtaining Historical Tesla Stock Data
# Downloads daily TESLA stock prices from Yahoo Finance for training & simulation
# Used yfinance instead of the .csv file provided by instructor because it is outdated (pre 2022 before the stock split) and will affect prediction error rate negatively
stock_data = yf.download("TSLA", start="2024-01-01", end="2025-03-29", auto_adjust=True) # Considered time period of 15 months (January 2024 to March 2025) for appropriate balance between the Random Forest model and the features chosen

# Feature Engineering
# Compute technical indicators to use as model inputs
# 5-day SMA (Simple Moving Average)
# 14-day RSI (Relative Strength Index)
stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Remove rows with NaN values caused by rolling indicators
stock_data.dropna(inplace=True)

# Prepare the Training Data using SMA and RSI
# Use current-day Close, SMA, RSI to predict next-day Close price
# Split into features (X) and labels (y)
features = stock_data[['Close', 'SMA_5', 'RSI']].copy()
features['Target'] = stock_data['Close'].shift(-1)
features.dropna(inplace=True)

X = features[['Close', 'SMA_5', 'RSI']]
y = features['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

# Train the Model (Random Forest Regressor)
# Use Random Forest to learn price movement patterns from historical data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model Accuracy using the MSE (Mean Square Error) since we are using a regression model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model MSE: {mse:.4f}")

# Strategy Logic
# Decide Buy/Sell/Hold based on predicted price movement
# A small threshold is used to avoid reacting to minor noise
def decide_action(predicted_price, current_price):
    diff = predicted_price - current_price
    threshold = 0.5
    if diff > threshold:
        return 'BUY'
    elif diff < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

# Simulate Daily Trading
# For each day:
# - Predict next day's price
# - Decide action
# - Execute trade logic
# - Track trades for chart and summary
buy_signals = []
sell_signals = []
dates_traded = []
prices_traded = []
actions = []

for date in TRADING_DAYS:
    if date not in stock_data.index:
        continue

    row = stock_data.loc[date]
    current_price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
    sma = float(row['SMA_5'].iloc[0]) if isinstance(row['SMA_5'], pd.Series) else float(row['SMA_5'])
    rsi = float(row['RSI'].iloc[0]) if isinstance(row['RSI'], pd.Series) else float(row['RSI'])

    X_live = pd.DataFrame([[current_price, sma, rsi]], columns=['Close', 'SMA_5', 'RSI'])
    predicted_price = model.predict(X_live)[0]

    action = decide_action(predicted_price, current_price)

    if action == 'BUY' and balance > 0:
        amount_to_spend = balance * 0.5
        shares_to_buy = (amount_to_spend * (1 - TRANSACTION_FEE)) / current_price
        balance -= amount_to_spend
        stocks_owned += shares_to_buy
        trade_log.append((date.date(), 'BUY', round(current_price, 2), round(shares_to_buy, 4), round(balance, 2)))
        buy_signals.append(current_price)
        sell_signals.append(np.nan)

    elif action == 'SELL' and stocks_owned > 0:
        shares_to_sell = stocks_owned * 0.5
        proceeds = shares_to_sell * current_price * (1 - TRANSACTION_FEE)
        balance += proceeds
        stocks_owned -= shares_to_sell
        trade_log.append((date.date(), 'SELL', round(current_price, 2), round(shares_to_sell, 4), round(balance, 2)))
        buy_signals.append(np.nan)
        sell_signals.append(current_price)

    else:
        trade_log.append((date.date(), 'HOLD', round(current_price, 2), 0, round(balance, 2)))
        buy_signals.append(np.nan)
        sell_signals.append(np.nan)

    dates_traded.append(date)
    prices_traded.append(current_price)
    actions.append(action)

# Final Balance Calculation
# Value = remaining cash + value of held shares at final stock price
last_valid_day = None
for date in reversed(TRADING_DAYS):
    if date in stock_data.index:
        last_valid_day = date
        break

if last_valid_day:
    final_price = float(stock_data.loc[last_valid_day]['Close'].iloc[0]) if isinstance(stock_data.loc[last_valid_day]['Close'], pd.Series) else float(stock_data.loc[last_valid_day]['Close'])
else:
    final_price = 0

portfolio_value = balance + (stocks_owned * final_price)

# Output the trading summary
print("\n===== TRADING SUMMARY =====")
for log in trade_log:
    print(f"{log[0]} | {log[1]} | Price: ${log[2]} | Shares: {log[3]} | Balance: ${log[4]}")

print("\n===== FINAL RESULTS =====")
print(f"Cash Balance: ${balance:.2f}")
print(f"Stocks Owned: {stocks_owned:.4f} shares")
print(f"Final Stock Price: ${final_price:.2f}")
print(f"Portfolio Value: ${portfolio_value:.2f}")
print(f"Net Profit: ${portfolio_value - STARTING_BALANCE:.2f}")

# Added a chart for visualization of multiple days
# Plot stock price over simulation period and mark Buy/Sell actions
plt.figure(figsize=(10, 6))
plt.plot(dates_traded, prices_traded, label='TSLA Close Price', marker='o')
plt.plot(dates_traded, buy_signals, '^', markersize=10, color='green', label='Buy Signal')
plt.plot(dates_traded, sell_signals, 'v', markersize=10, color='red', label='Sell Signal')
plt.title('TSLA Trading Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()