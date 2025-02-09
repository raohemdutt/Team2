import pandas as pd
df = pd.read_csv("indicators_and_data.csv")
print(df.head())
# Define breakout conditions
df["Bullish_Breakout"] = (df["Close"] > df["BB_High"]) & (df["Volume"] > 1.5 * df["Vol_20"])
df["Bearish_Breakout"] = (df["Close"] < df["BB_Low"]) & (df["Volume"] > 1.5 * df["Vol_20"])

# Count total breakouts per asset
breakout_counts = df.groupby("Symbol")[["Bullish_Breakout", "Bearish_Breakout"]].sum()

# Add a total breakout column
breakout_counts["Total_Breakouts"] = breakout_counts["Bullish_Breakout"] + breakout_counts["Bearish_Breakout"]

# Display the breakout counts per asset
print(breakout_counts.sort_values(by="Total_Breakouts", ascending=True))
