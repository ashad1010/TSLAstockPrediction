import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configuration
STARTING_BALANCE = 10000
TRANSACTION_FEE = 0.01  # 1%
TRADING_DAYS = pd.date_range(start="2025-03-10", end="2025-03-14", freq='B')  # Change this for demo or real week

# Initialize state
balance = STARTING_BALANCE
stocks_owned = 0
trade_log = []

# Download historical Tesla data
stock_data = yf.download("TSLA", start="2025-01-01", end="2025-03-29", auto_adjust=True)

# Calculate indicators
stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Strategy Logic
def decide_action(rsi, close, sma):
    if pd.isna(rsi) or pd.isna(sma):
        return 'HOLD'
    if rsi < 30 and close > sma:
        return 'BUY'
    elif rsi > 70 and close < sma:
        return 'SELL'
    else:
        return 'HOLD'

# Simulate Trading
for date in TRADING_DAYS:
    if date not in stock_data.index:
        continue

    row = stock_data.loc[date]

    # Extract clean values
    rsi = float(row['RSI'])
    close = float(row['Close'])
    sma = float(row['SMA_5'])
    price = close  # price is now guaranteed to be defined

    action = decide_action(rsi, close, sma)

    if action == 'BUY' and balance > 0:
        amount_to_spend = balance * 0.5  # Spend 50% of balance
        shares_to_buy = (amount_to_spend * (1 - TRANSACTION_FEE)) / price
        balance -= amount_to_spend
        stocks_owned += shares_to_buy
        trade_log.append((date.date(), 'BUY', round(price, 2), round(shares_to_buy, 4), round(balance, 2)))

    elif action == 'SELL' and stocks_owned > 0:
        shares_to_sell = stocks_owned * 0.5  # Sell 50% of holdings
        proceeds = shares_to_sell * price * (1 - TRANSACTION_FEE)
        balance += proceeds
        stocks_owned -= shares_to_sell
        trade_log.append((date.date(), 'SELL', round(price, 2), round(shares_to_sell, 4), round(balance, 2)))

    else:
        trade_log.append((date.date(), 'HOLD', round(price, 2), 0, round(balance, 2)))

# Final Balance Calculation
last_valid_day = None
for date in reversed(TRADING_DAYS):
    if date in stock_data.index:
        last_valid_day = date
        break

if last_valid_day:
    final_price = float(stock_data.loc[last_valid_day]['Close'])
else:
    final_price = 0

portfolio_value = balance + (stocks_owned * final_price)

# Output summary
print("\n===== TRADING SUMMARY =====")
for log in trade_log:
    print(f"{log[0]} | {log[1]} | Price: ${log[2]} | Shares: {log[3]} | Balance: ${log[4]}")

print("\n===== FINAL RESULTS =====")
print(f"Cash Balance: ${balance:.2f}")
print(f"Stocks Owned: {stocks_owned:.4f} shares")
print(f"Final Stock Price: ${final_price:.2f}")
print(f"Portfolio Value: ${portfolio_value:.2f}")
print(f"Net Profit: ${portfolio_value - STARTING_BALANCE:.2f}")
