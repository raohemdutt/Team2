import yfinance as yf
import pandas as pd
# Define the date range for historical interest rate data
start_date = "2010-06-07"
end_date = "2025-01-21"

# Define interest rate tickers available on Yahoo Finance
interest_rate_tickers = {
    "FedFundsRate": "^IRX",  # 13-week Treasury Bill Rate (proxy for Fed Funds Rate)
    "US10Y": "^TNX",         # 10-Year Treasury Yield
    "US30Y": "^TYX",         # 30-Year Treasury Yield
}

# Fetch historical data for the specified date range
interest_rate_data = {}
for name, ticker in interest_rate_tickers.items():
    rate = yf.Ticker(ticker)
    hist = rate.history(start=start_date, end=end_date)
    if not hist.empty:
        interest_rate_data[name] = hist["Close"]  # Extract closing values

# Combine data into a single DataFrame
rates_df = pd.DataFrame(interest_rate_data)
rates_df.index = pd.to_datetime(rates_df.index).date  # Convert index to date format

# Save to CSV for integration with the strategy
csv_path = "interest_rates.csv"
rates_df.to_csv(csv_path)

