import pandas as pd
df = pd.read_csv("indicators_and_data.csv")
print(df.head())
# Ensure necessary columns exist
required_columns = ["Symbol", "Close", "BB_High", "BB_Low", "ATR_14", "RSI_14"]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns in the dataset.")

# Drop rows with missing values in key columns
df = df.dropna(subset=required_columns)

# Compute the 75th percentile of ATR_14 beforehand
atr_threshold = df["ATR_14"].quantile(0.75)

# Define breakout detection with reset mechanism
def detect_breakout(group):
    breakout_triggered = False  # Ensures a breakout must reset before triggering another
    breakout_counts = []

    for index, row in group.iterrows():
        breakout = 0
        
        if not breakout_triggered:
            if row["Close"] > row["BB_High"]:  # Bullish breakout
                breakout = 1
                breakout_triggered = True  # Prevent further signals until reset
            elif row["Close"] < row["BB_Low"]:  # Bearish breakout
                breakout = 1
                breakout_triggered = True  # Prevent further signals until reset

            # Additional conditions
            if row["ATR_14"] > atr_threshold:  # High volatility breakout
                breakout += 1
            if row["RSI_14"] > 70 or row["RSI_14"] < 30:  # Overbought/Oversold breakout
                breakout += 1
        
        if breakout > 0:
            breakout_triggered = True  # Prevents another breakout from being counted immediately
        else:
            breakout_triggered = False  # Resets breakout state only after no signal

        breakout_counts.append((index, breakout))  # Store index to preserve alignment

    return pd.DataFrame(breakout_counts, columns=["Index", "Breakout_Count"]).set_index("Index")

# Apply breakout detection by asset with include_groups=False
breakout_results = df.groupby("Symbol", group_keys=False).apply(detect_breakout).reset_index()

# Merge results back to the main dataframe correctly
df = df.merge(breakout_results, left_index=True, right_index=True, how="left")

# Count breakouts per asset
breakout_summary = df.groupby("Symbol")["Breakout_Count"].sum().reset_index()

# Get top 5 assets with the most breakouts
top_breakouts = breakout_summary.sort_values(by="Breakout_Count", ascending=False).head(5)

# Display results
print(top_breakouts)
