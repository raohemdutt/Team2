import pandas as pd

# ✅ Load market & interest rate data
df = pd.read_csv("indicators_and_data.csv")  # Market data
rates_df = pd.read_csv("interest_rates.csv")  # Interest rate dataset

# # ✅ Convert Date column to datetime
# df["Date"] = pd.to_datetime(df["Date"])
# rates_df["Date"] = pd.to_datetime(rates_df["Date"])

df["Date"] = pd.to_datetime(df["Date"]).dt.date  # Extract only the date (ignore time zone)
rates_df["Date"] = pd.to_datetime(rates_df["Date"]).dt.date  # Extract only the date

rates_df = rates_df.sort_values(by="Date", ascending=False).reset_index(drop=True)

# ✅ Merge interest rates into the main dataset
df = df.merge(rates_df, on="Date", how="left")

# ✅ Hardcoded Asset-Sector Mapping
asset_sector_map = {
    "6B": "FX", "6C": "FX", "6E": "FX", "6J": "FX", "6M": "FX", "6N": "FX", "6S": "FX",
    "CL": "Energy", "RB": "Energy",
    "GC": "Metals", "SI": "Metals", "HG": "Metals", "PL": "Metals",
    "GF": "Agriculture", "HE": "Agriculture", "LE": "Agriculture", "KE": "Agriculture",
    "ZC": "Agriculture", "ZL": "Agriculture", "ZM": "Agriculture", "ZR": "Agriculture", "ZS": "Agriculture", "ZW": "Agriculture",
    "MES": "Equities", "MNQ": "Equities", "MYM": "Equities", "RTY": "Equities",
    "ZN": "Interest Rate", "UB": "Interest Rate",
}

# ✅ Assign Sectors to Assets
df["Sector"] = df["Symbol"].map(asset_sector_map)


# ✅ Compute Interest Rate Trends (Rolling Change Over 10 Days)
df["Short_Rate_Trend"] = df["FedFundsRate"].diff(10).fillna(0)  # Fed Funds Rate (Short-Term)
df["Medium_Rate_Trend"] = df["US10Y"].diff(10).fillna(0)        # 10-Year Treasury (Medium-Term)
df["Long_Rate_Trend"] = df["US30Y"].diff(10).fillna(0)          # 30-Year Treasury (Long-Term)

# ✅ Assign Trade Preference Based on Interest Rate Environment
df["Rate_Favored"] = "Neutral"

# Rising Short-Term Rates: Favor financials, commodities, energy
df.loc[(df["Short_Rate_Trend"] > 0), "Rate_Favored"] = "Short-Term Rising"

# Rising Medium-Term Rates: Favor value stocks, bonds may underperform
df.loc[(df["Medium_Rate_Trend"] > 0), "Rate_Favored"] = "Medium-Term Rising"

# Rising Long-Term Rates: Favor cyclical sectors (industrials, energy)
df.loc[(df["Long_Rate_Trend"] > 0), "Rate_Favored"] = "Long-Term Rising"

# Falling Short-Term Rates: Favor rate-sensitive stocks (tech, bonds, real estate)
df.loc[(df["Short_Rate_Trend"] < 0), "Rate_Favored"] = "Short-Term Falling"

# Falling Medium/Long-Term Rates: Favor bonds, high-growth equities, real estate
df.loc[(df["Medium_Rate_Trend"] < 0) | (df["Long_Rate_Trend"] < 0), "Rate_Favored"] = "Long-Term Falling"

# ✅ Adjust Volume Anomaly Thresholds Based on Liquidity Conditions
df["Adjusted_Vol_Threshold"] = df["Vol_20"]
df.loc[df["Rate_Favored"] == "Short-Term Rising", "Adjusted_Vol_Threshold"] *= 1.2  # Higher threshold in rising rates
df.loc[df["Rate_Favored"] == "Short-Term Falling", "Adjusted_Vol_Threshold"] *= 0.8  # Lower threshold in falling rates
df.loc[df["Rate_Favored"] == "Long-Term Falling", "Adjusted_Vol_Threshold"] *= 0.75  # Lower threshold in long-term falling

# ✅ Define Breakout Conditions Based on Rate Sensitivity
df["Bullish_Breakout"] = False
df["Bearish_Breakout"] = False

for i in range(len(df)):
    if df.loc[i, "Rate_Favored"] == "Short-Term Rising":
        df.loc[i, "Bullish_Breakout"] = (
            (df.loc[i, "Close"] > df.loc[i, "BB_High"]) &
            (df.loc[i, "Volume"] > 1.5 * df.loc[i, "Adjusted_Vol_Threshold"]) &
            (df.loc[i, "Sector"] in ["Financials", "Energy", "Commodities"])
        )
    elif df.loc[i, "Rate_Favored"] in ["Medium-Term Rising", "Long-Term Rising"]:
        df.loc[i, "Bullish_Breakout"] = (
            (df.loc[i, "Close"] > df.loc[i, "BB_High"]) &
            (df.loc[i, "Volume"] > 1.5 * df.loc[i, "Adjusted_Vol_Threshold"]) &
            (df.loc[i, "Sector"] in ["Value Stocks", "Industrials", "Materials"])
        )
    elif df.loc[i, "Rate_Favored"] == "Short-Term Falling":
        df.loc[i, "Bearish_Breakout"] = (
            (df.loc[i, "Close"] < df.loc[i, "BB_Low"]) &
            (df.loc[i, "Volume"] > 1.5 * df.loc[i, "Adjusted_Vol_Threshold"]) &
            (df.loc[i, "Sector"] in ["Bonds", "Real Estate", "Tech"])
        )
    elif df.loc[i, "Rate_Favored"] in ["Long-Term Falling"]:
        df.loc[i, "Bullish_Breakout"] = (
            (df.loc[i, "Close"] > df.loc[i, "BB_High"]) &
            (df.loc[i, "Volume"] > 1.5 * df.loc[i, "Adjusted_Vol_Threshold"]) &
            (df.loc[i, "Sector"] in ["Growth Stocks", "Tech", "Real Estate", "Bonds"])
        )
    else:
        # Default volume-based breakouts (neutral environment)
        df.loc[i, "Bullish_Breakout"] = (
            (df.loc[i, "Close"] > df.loc[i, "BB_High"]) &
            (df.loc[i, "Volume"] > 1.5 * df.loc[i, "Adjusted_Vol_Threshold"])
        )
        df.loc[i, "Bearish_Breakout"] = (
            (df.loc[i, "Close"] < df.loc[i, "BB_Low"]) &
            (df.loc[i, "Volume"] > 1.5 * df.loc[i, "Adjusted_Vol_Threshold"])
        )

# ✅ Count Total Breakouts per Asset
breakout_counts = df.groupby("Symbol")[["Bullish_Breakout", "Bearish_Breakout"]].sum()

# ✅ Add a Total Breakout Column
breakout_counts["Total_Breakouts"] = breakout_counts["Bullish_Breakout"] + breakout_counts["Bearish_Breakout"]

# ✅ Save Updated Asset Selection Data
df.to_csv("updated_asset_selection.csv", index=False)

# ✅ Display the Breakout Counts per Asset
print(breakout_counts.sort_values(by="Total_Breakouts", ascending=True))
