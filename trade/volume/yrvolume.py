import yfinance as yf
import pandas as pd
import datetime

# Define sector mappings using representative ETFs
sector_etfs = {
    "Interest Rate": "TLT",  # Treasury Bonds ETF
    "Agriculture": "DBA",    # Agriculture ETF
    "Metals": "GLD",         # Gold ETF
    "Equities": "SPY",       # S&P 500 ETF
    "Energy": "XLE",         # Energy ETF
    "FX": "UUP"             # US Dollar Index ETF
}

# Define start and end year for historical data
start_year = 2010
end_year = 2025

# Dictionary to store yearly average volume data
yearly_sector_volume = []

# Fetch historical data for each sector ETF
for sector, ticker in sector_etfs.items():
    stock = yf.Ticker(ticker)
    hist = stock.history(period="15y")  # Get max available data (adjust later per year)
    
    # Ensure the Date column is in datetime format
    hist.index = pd.to_datetime(hist.index)

    for year in range(start_year, end_year + 1):
        # Filter data for the specific year
        yearly_data = hist.loc[hist.index.year == year]
        if not yearly_data.empty:
            avg_volume = yearly_data["Volume"].mean()
            yearly_sector_volume.append({"Year": year, "Sector": sector, "YearlyAverageVolume": avg_volume})

# Convert to DataFrame
sector_volume_df = pd.DataFrame(yearly_sector_volume)

# Save to CSV
sector_volume_df.to_csv("trade/volume/yearly_sector_volume_2010_2025.csv", index=False)