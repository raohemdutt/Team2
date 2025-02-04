# import pandas as pd
# import numpy as np

# # === Generate Dummy Asset Data for Multiple Assets === #
# def generate_multi_asset_data(start_date="2024-01-01", num_days=100, assets=["AAPL", "TSLA", "GOOGL"]):
#     """
#     Generates simulated OHLC price and volume data for multiple assets.
#     """
#     dates = pd.date_range(start=start_date, periods=num_days)
#     data = []
    
#     for asset in assets:
#         open_prices = np.cumsum(np.random.randn(num_days) * 2 + 100)
#         high_prices = open_prices + np.random.rand(num_days) * 2
#         low_prices = open_prices - np.random.rand(num_days) * 2
#         close_prices = open_prices + np.random.randn(num_days) * 1.5
#         volumes = np.random.randint(50000, 200000, size=num_days)
#         lookback = [20] * num_days  # Default lookback period
        
#         for i in range(num_days):
#             data.append([dates[i], asset, open_prices[i], high_prices[i], low_prices[i], close_prices[i], volumes[i], lookback[i]])
    
#     df = pd.DataFrame(data, columns=["Date", "Asset", "Open", "High", "Low", "Close", "Volume", "Lookback"])
    
#     return df

# # === Generate Dummy Market Volume Data === #
# def generate_market_volume_data(start_date="2024-01-01", num_days=100, sectors=["Tech", "Energy"]):
#     """
#     Generates simulated market-wide volume data for different sectors.
#     """
#     dates = pd.date_range(start=start_date, periods=num_days)
#     data = []
    
#     for sector in sectors:
#         market_volumes = np.random.randint(90000000, 120000000, size=num_days)  # Simulated market volume
        
#         for i in range(num_days):
#             data.append([dates[i], sector, market_volumes[i]])
    
#     df = pd.DataFrame(data, columns=["Date", "Sector", "MarketVolume"])
    
#     return df

# # === Save Generated Data to CSV Files === #
# def save_dummy_data():
#     """
#     Generates and saves multi-asset data and market volume data to CSV files.
#     """
#     asset_df = generate_multi_asset_data()
#     market_df = generate_market_volume_data()
    
#     asset_df.to_csv("asset_data.csv", index=False)
#     market_df.to_csv("market_volume.csv", index=False)
    
#     print("Dummy data files generated: asset_data.csv, market_volume.csv")

# # === Run Dummy Data Generation === #
# if __name__ == "__main__":
#     save_dummy_data()


import pandas as pd
import numpy as np

# Define the structure of the dataset
columns = ["Date", "Asset", "Open", "High", "Low", "Close", "Volume", "Lookback", "Sector", "SectorMarketVolume"]

# Generate random data
num_days = 100  # Extend dataset to 100 days
assets = ["ES", "CL", "GC", "NQ", "ZN"]  # Example asset symbols
sectors = {"ES": "Index", "CL": "Energy", "GC": "Metals", "NQ": "Tech", "ZN": "Bonds"}

# Generate data
data = []
start_date = pd.Timestamp("2024-01-01")

for i in range(num_days):
    date = start_date + pd.Timedelta(days=i)
    for asset in assets:
        open_price = np.round(np.random.uniform(100, 300), 2)
        high_price = np.round(open_price + np.random.uniform(0.5, 3), 2)
        low_price = np.round(open_price - np.random.uniform(0.5, 3), 2)
        close_price = np.round(np.random.uniform(low_price, high_price), 2)
        volume = np.random.randint(50000, 150000)
        lookback = 20
        sector = sectors[asset]
        sector_market_volume = np.round(close_price + np.random.uniform(0, 5), 2)

        data.append([date.strftime("%Y-%m-%d"), asset, open_price, high_price, low_price, close_price, volume, lookback, sector, sector_market_volume])

# Create DataFrame
df_random = pd.DataFrame(data, columns=columns)

# Save to CSV
df_random.to_csv("asset_data.csv", index=False)
