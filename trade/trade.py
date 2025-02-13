import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multipliers import portfolio_risk_aggregator
import datetime

# === Trading Logic Implementation === #
# This script gets preprocessed data (Open, High, Low, Close, Volume) is provided from a CSV.
# It implements dynamic reward-to-risk, stop-loss adaptation, and volume-based position sizing.

def load_futures_specifications(csv_file="trade/futures_specifications.csv"):
    """
    Loads futures contract specifications from a CSV file.
    """
    df = pd.read_csv(csv_file)  # Read CSV
    futures_dict = {}

    for _, row in df.iterrows():
        futures_dict[row["Ticker"]] = {
            "tick_size": row["TickSize"],
            "tick_value": row["TickValue"],
            "contract_size": row["ContractSize"],
            "margin_requirement": row["MarginRequirement"],
            # "expiry_date": row["ExpiryDate"]
            "rollover_months": [m.strip() for m in row["RolloverMonths"].split(',')]  # Store months as a list
        }

    return futures_dict

# Load specifications
FUTURES_SPECIFICATIONS = load_futures_specifications()


# === Function to Calculate Indicators (DRI, VAS, ATR) === #
def calculate_indicators(df):
    """
    Calculate key trading indicators for each asset separately.
    - DRI: Measures daily price range compared to historical averages.
    - VAS: Volume anomaly score compares current volume to rolling average.
    - ATR: Average True Range helps set dynamic stop-loss levels.
    """
    assets = df["Asset"].unique()
    all_data = []
    
    for asset in assets:
        asset_df = df[df["Asset"] == asset].copy()
        asset_df['Range'] = asset_df['High'] - asset_df['Low']
       
       
                # Apply rolling window dynamically using Lookback column
        # asset_df['ATR'] = asset_df.apply(lambda row: 
        #                                  asset_df.loc[:row.name, 'Range'].rolling(window=row['Lookback']).mean().iloc[-1], axis=1)
        asset_df['ATR'] = asset_df.apply(lambda row: 
            asset_df.loc[:row.name, 'Range'].rolling(window=row['Lookback']).mean().shift(1).iloc[-1], axis=1)

        asset_df['DRI'] = asset_df.apply(lambda row: 
                                         row['Range'] / asset_df.loc[:row.name, 'Range'].rolling(window=row['Lookback']).mean().shift(1).iloc[-1], axis=1)

        asset_df['AvgVolume'] = asset_df.apply(lambda row: 
                                               asset_df.loc[:row.name, 'Volume'].rolling(window=row['Lookback']).mean().shift(1).iloc[-1], axis=1)

        asset_df['VAS'] = asset_df['Volume'] / asset_df['AvgVolume']
        
        all_data.append(asset_df)

        # print(df[['Asset', 'Lookback', 'ATR', 'VAS', 'DRI']].dropna().head())

    
    return pd.concat(all_data, ignore_index=True)

# === Function to Identify Consolidation and Breakouts === #
def identify_breakouts(df):
    """
    Identify breakout opportunities per asset.
    - Detects price consolidation phases using DRI.
    - Determines bullish or bearish breakout signals based on volume (VAS) and price movement.
    """
    assets = df["Asset"].unique()
    all_data = []
    
    for asset in assets:
        asset_df = df[df["Asset"] == asset].copy()
                # Ensure VIR exists in the filtered dataframe
        if 'VIR' not in asset_df.columns:
            raise KeyError(f"VIR column is missing in asset_df for asset {asset}. Check integration step.")

        # Consolidation Detection
        asset_df['Consolidation'] = asset_df['DRI'].rolling(window=3, min_periods=1).apply(lambda x: all(x < 0.8), raw=True).astype(bool)

        # Consolidation High/Low using forward fill
        asset_df['ConsolidationHigh'] = asset_df['High'].where(asset_df['Consolidation']).ffill()
        asset_df['ConsolidationLow'] = asset_df['Low'].where(asset_df['Consolidation']).ffill()

        # Breakout Signals
        asset_df['BullishBreakout'] = (
            (asset_df['Close'] > asset_df['ConsolidationHigh']) &
            (asset_df['VAS'] > 1.5) & 
            (asset_df['VIR'] > 1.2)  # Ensuring stock-specific volume anomaly
        )

        asset_df['BearishBreakout'] = (
            (asset_df['Close'] < asset_df['ConsolidationLow']) &
            (asset_df['VAS'] > 1.5) & 
            (asset_df['VIR'] > 1.2)  # Ensuring stock-specific volume anomaly
        )
        
        all_data.append(asset_df)

        print(asset_df[['Close', 'ConsolidationHigh', 'ConsolidationLow', 'VAS']].dropna().head())

    return pd.concat(all_data, ignore_index=True)

# === Function to Integrate Yearly Sector Volume === #
def integrate_sector_volume(asset_df, sector_volume_df):
    """
    Merges yearly sector volume data into asset data by matching sector and year.
    Updates asset_data.csv immediately after merging.
    """
    asset_df["Date"] = pd.to_datetime(asset_df["Date"])
    asset_df["Year"] = asset_df["Date"].dt.year
    
    # Merge sector volume data
    asset_df = asset_df.merge(sector_volume_df, on=["Sector", "Year"], how="left")
    asset_df.rename(columns={"YearlyAverageVolume": "SectorMarketVolumes"}, inplace=True)
    
    # Ensure the column exists
    if "SectorMarketVolumes" not in asset_df.columns:
        asset_df["SectorMarketVolumes"] = 0  # Default to zero if missing
    
    # Save updated asset data immediately
    asset_df.to_csv("trade/asset_data.csv", index=False)
    print("Updated asset_data.csv with sector volume.")
    return asset_df


# === Function to Integrate Volume Index Comparison === #
def integrate_volume_index(df):
    """
    Compare asset volume to a broader market volume benchmark.
    - Ensures each asset is compared to its corresponding market sector.
    """

    # Compute Market Volume Anomaly Score (MarketVAS)
    df['MarketAvgVolume'] = df.groupby('Sector')['SectorMarketVolume'].transform(lambda x: x.rolling(window=df['Lookback'].iloc[0]).mean())
    df['MarketVAS'] = df['SectorMarketVolume'] / df['MarketAvgVolume']

    df['VIR'] = df['VAS'] / df['MarketVAS']  # Volume Index Ratio
    
    return df

# === Function to Adjust Stop-Loss & Take-Profit Dynamically === #
def adjust_stop_loss_target(df):
    """
    Set stop-loss and target prices dynamically per asset.
    - Uses ATR to determine stop-loss levels.
    - Dynamically adjusts reward-to-risk ratios.
    """
    assets = df["Asset"].unique()
    all_data = []
    
    for asset in assets:
        asset_df = df[df["Asset"] == asset].copy()
        
                # Reduce stop-loss range dynamically if price is consolidating
        asset_df['ATR_Stop_Adjust'] = np.where(asset_df['Consolidation'], 0.5 * asset_df['ATR'], asset_df['ATR'])
        asset_df['StopLoss'] = np.where(asset_df['BullishBreakout'],
                                        asset_df['Close'] - (0.3 * asset_df['ATR_Stop_Adjust']),
                                        asset_df['Close'] + (0.3 * asset_df['ATR_Stop_Adjust']))

        # asset_df['RewardToRisk'] = (asset_df['ATR'] / (asset_df['ConsolidationHigh'] - asset_df['ConsolidationLow'])).clip(1.5, 3.0)
        # Define true risk level based on stop-loss distance
        # Define true risk based on stop-loss distance
        true_risk = np.abs(asset_df['Close'] - asset_df['StopLoss'])

        # Estimate realistic reward using past breakouts
        historical_max_move = asset_df['High'].rolling(window=20).max() - asset_df['Close']
        historical_min_move = asset_df['Close'] - asset_df['Low'].rolling(window=20).min()

        # Use past breakout volatility as a proxy for potential reward
        estimated_reward = np.where(asset_df['BullishBreakout'], historical_max_move, historical_min_move)

        # Calculate Reward-to-Risk (RR) dynamically
        asset_df['RewardToRisk'] = (estimated_reward / true_risk).clip(1.5, 3.0)

        asset_df['Target'] = np.where(asset_df['BullishBreakout'],
                                      asset_df['Close'] + asset_df['RewardToRisk'] * (asset_df['Close'] - asset_df['StopLoss']),
                                      asset_df['Close'] - asset_df['RewardToRisk'] * (asset_df['StopLoss'] - asset_df['Close']))
        
        all_data.append(asset_df)
    
    return pd.concat(all_data, ignore_index=True)


def calculate_risk_matrices(df):
    """
    Computes:
    1. Covariance matrix (for normal market volatility risk)
    2. Jump risk covariance matrix (for extreme price movements)
    """
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

    # Create returns matrix (each column is an asset's returns)
    returns_matrix = df.pivot(index='Date', columns='Asset', values='LogReturn')

    # Covariance matrix for normal volatility
    covariance_matrix = returns_matrix.cov()

    # Identify extreme price jumps (returns > 2 standard deviations)
    jump_threshold = returns_matrix.std() * 2
    jump_returns = returns_matrix.copy()
    jump_returns = jump_returns[(jump_returns.abs() > jump_threshold)]

    # Jump covariance matrix (only extreme movements)
    jump_covariance_matrix = jump_returns.cov()

    return covariance_matrix, jump_covariance_matrix

def handle_futures_rollover(df):
    """
    Rolls over futures contracts based on available months instead of fixed expiry dates.
    The rollover date is determined based on the date of the trade or data being considered.
    """
    for asset, specs in FUTURES_SPECIFICATIONS.items():
        rollover_months = specs['rollover_months']
        # print(f"Processing asset: {asset}, Rollover Months: {rollover_months}")  # Debug asset and rollover months

        for index, row in df[df['Asset'] == asset].iterrows():
            trade_date = pd.Timestamp(row['Date'])
            trade_month = trade_date.strftime('%b')
            # print(f"Trade Date: {trade_date.date()}, Trade Month: {trade_month}")  # Debug trade date

            # Find the next available month for rollover
            next_rollover_month = None
            for month in rollover_months:
                if datetime.datetime.strptime(month, '%b').month > trade_date.month:
                    next_rollover_month = month
                    # print(f"Next Rollover Month Found: {next_rollover_month}")  # Debug next rollover month

                    break
            
            # If no future month is found, roll over to the earliest available month
            if not next_rollover_month:
                next_rollover_month = rollover_months[0]  # Default to the first month in list
                # print(f"No future month found. Rolling over to first available month: {next_rollover_month}")

            df.at[index, 'Rollover'] = True  # Flag trades for rollover
            # print(f"Rollover flag set for index {index}")  # Debug index of updated row
    print("Final DataFrame:\n", df.head())  # Print first few rows of updated DataFrame

    
    return df

def calculate_futures_transaction_cost(trade):
    """
    Computes realistic transaction costs for futures trading.
    """
    asset = trade['Asset']
    if asset in FUTURES_SPECIFICATIONS:
        spec = FUTURES_SPECIFICATIONS[asset]
        contracts = trade['Contracts']
        
        # Exchange & broker fees per contract (assumed)
        exchange_fee = 1.50  # CME fee per contract
        broker_fee = 2.00  # Broker commission per contract
        
        # Total transaction cost per contract
        total_cost = contracts * (exchange_fee + broker_fee) * 2  # Multiply by 2 (entry & exit)

        return total_cost
    return 0  # No cost for non-futures trades

def calculate_futures_pnl(trades):
    """
    Adjusts P&L calculations to use futures contract specifications.
    """
    if isinstance(trades, list) and not trades:  # Handle empty trade list
        return pd.DataFrame(columns=['EntryDate', 'ExitDate', 'Asset', 'EntryPrice', 'ExitPrice', 'GrossPnL', 'NetPnL'])
    
    trades = pd.DataFrame(trades)  # Ensure trades is a DataFrame

    # print("Debug - Checking Contracts Column:")
    # print(trades[['Asset', 'Contracts']].dropna().head())  # Ensure Contracts exist
    Grss_pnl=0
    Nt_pnl=0
    for i, trade in trades.iterrows():
        asset = trade['Asset']
        if asset in FUTURES_SPECIFICATIONS:
            spec = FUTURES_SPECIFICATIONS[asset]
            tick_size, tick_value, contract_size = spec['tick_size'], spec['tick_value'], spec['contract_size']

            # Compute P&L using futures-specific calculations
            if trade['Direction'] == 'Bearish':
                price_difference = trade['EntryPrice'] - trade['ExitPrice']
            elif trade['Direction'] == 'Bullish':
                price_difference = trade['ExitPrice'] - trade['EntryPrice']

            # print("price difference",price_difference)
            tick_movement = price_difference / tick_size  # Number of ticks moved
            # print("tick movement", tick_movement)
            gross_pnl = tick_movement * tick_value * trade['Contracts']

            # Apply transaction costs (per contract per trade)
            total_transaction_cost = calculate_futures_transaction_cost(trade)  # 

            # Compute net P&L
            net_pnl = gross_pnl - total_transaction_cost

            # Update trade DataFrame
            trades.at[i, 'GrossPnL'] = gross_pnl
            Grss_pnl+=gross_pnl
            trades.at[i, 'TransactionCost'] = total_transaction_cost

            trades.at[i, 'NetPnL'] = net_pnl
            Nt_pnl+=net_pnl
    print("GROSS PNL",Grss_pnl)
    print("NET PNL",Nt_pnl)

    return trades

# === Function to Calculate Dynamic Position Sizing === #
def calculate_position_size(df, capital=500000, risk_per_trade=0.01, atr_multiplier=1.5, max_leverage=2.0, 
                            max_correlation_risk=0.2, max_portfolio_volatility=0.15, max_jump_risk=0.1, max_sector_exposure=0.2):
    """
    Determines position size per asset using:
    1. ATR-based volatility scaling (reduces size in high-volatility markets)
    2. VAS-based volume scaling (prioritizes high-confidence setups)

    - Ensures dynamic risk-adjusted position sizing.
    - Balances volatility risk and volume-based opportunities.
    """
    assets = df["Asset"].unique()
    all_data = []
    positions_list = []  # Stores per-asset positions
    total_margin_used = 0

    for asset in assets:
        asset_df = df[df["Asset"] == asset].copy()
        risk_amount = capital * risk_per_trade  # Risk allocated per trade
        
        # Volatility-adjusted risk (higher ATR = smaller position size)
        asset_df['ATR_Adjusted_Size'] = risk_amount / (asset_df['ATR'] * atr_multiplier)

        # Apply volume-based scaling (VAS ensures high-confidence setups get more allocation)
        asset_df['PositionSize'] = asset_df['ATR_Adjusted_Size'] * np.minimum(asset_df['VAS'].fillna(0), 3)

        # FUTURES-SPECIFIC POSITION SIZING
        if asset in FUTURES_SPECIFICATIONS:
            contract_spec = FUTURES_SPECIFICATIONS[asset]
            contract_size = contract_spec['contract_size']
            tick_value = contract_spec['tick_value']
            margin_requirement = contract_spec['margin_requirement']

            asset_df['PositionSize'] = asset_df['PositionSize'].fillna(0)

            # Compute Contracts
            # contract_value = contract_size * tick_value
            contract_value = contract_size * asset_df['Close']  # Use price level instead of tick value

            asset_df['Contracts'] = asset_df['PositionSize'] / contract_value

            # Ensure contracts are rounded up to maximize profit since we already have position protections (no partial contracts)
            asset_df['Contracts'] = np.ceil(asset_df['Contracts']).astype(int)

            for i, row in asset_df.iterrows():
                position_margin = row['Contracts'] * margin_requirement
                if total_margin_used + position_margin > max_leverage * capital:
                    # Scale down contracts if margin exceeds limit
                    asset_df.at[i, 'Contracts'] = max(1, (max_leverage * capital - total_margin_used) // margin_requirement)
                total_margin_used += asset_df.at[i, 'Contracts'] * margin_requirement

            #  Apply Sector Balancing to Contracts
            sector_allocations = asset_df.groupby("Sector")["Contracts"].sum()
            for sector in sector_allocations.index:
                sector_weight = sector_allocations[sector] / capital  

                if sector_weight > max_sector_exposure:
                    reduction_factor = 1 - ((sector_weight - max_sector_exposure) / sector_weight)
                    reduction_factor = max(reduction_factor, 0.5)  # Prevent over-adjustment

                    print(f"Sector {sector} overweight! Applying reduction factor: {reduction_factor:.2f}")
                    asset_df.loc[asset_df["Sector"] == sector, "Contracts"] = np.floor(asset_df.loc[asset_df["Sector"] == sector, "Contracts"] * reduction_factor).astype(int)



        # Store positions for risk aggregation
        positions_list.append(asset_df['Contracts'].values)
        all_data.append(asset_df)

    df = pd.concat(all_data, ignore_index=True)

    covariance_matrix, jump_covariance_matrix = calculate_risk_matrices(df)

    # print("Debug - Positions list lengths:", [len(pos) for pos in positions_list])

    # Ensure positions have the same length as the DataFrame
    max_length = len(df)

    # Handle varying lengths by padding with the last valid value
    positions = np.array([
        np.pad(pos, (0, max_length - len(pos)), constant_values=pos[-1] if len(pos) > 0 else 0)[:max_length]
        for pos in positions_list
    ])

    # print(f"Debug - Positions shape: {positions.shape}, Expected DF Length: {len(df)}")

    # Compute positions weighted per asset
    unique_assets = df["Asset"].unique()

    # Compute positions weighted per asset
    positions_weighted = np.mean(positions, axis=1) / capital

    # Ensure positions_weighted is correctly shaped
    positions_weighted = np.array(positions_weighted).flatten()


    for i, asset in enumerate(unique_assets):
        asset_positions = positions[i, :]  # Extract positions for this asset
        mean_position = np.mean(asset_positions)
        positions_weighted[i] = mean_position / capital

    # Apply Carver's Risk Multipliers
    final_positions = portfolio_risk_aggregator(
        positions=positions,
        positions_weighted=positions_weighted.flatten(),
        covariance_matrix=covariance_matrix.to_numpy(),
        jump_covariance_matrix=jump_covariance_matrix.to_numpy(),
        maximum_portfolio_leverage=max_leverage,
        maximum_correlation_risk=max_correlation_risk,
        maximum_portfolio_risk=max_portfolio_volatility,
        maximum_jump_risk=max_jump_risk,
        date=pd.Timestamp.today()
    )

    # print(f"Debug - Final Positions shape: {final_positions.shape}, Expected DF Length: {len(df)}")

    # Ensure final_positions matches `df`
    if final_positions.shape[1] == len(df):  # If final_positions is (Assets, Time Steps)
        final_positions = final_positions.mean(axis=0)  # Take mean across assets

    # print(f"Debug - Final Positions shape AFTER FIX: {final_positions.shape}, Expected: ({len(df)},)")

    df['PositionSize'] = final_positions
    df['Contracts'] = final_positions

    return df

# === Function to Simulate Trades === #
def simulate_trades(df, exit_days=5, transaction_cost_pct=0.001, slippage_pct=0.0005, max_trades_per_day=10):
    """
    Execute trade simulation per asset.
    - Identifies trade entries based on breakout conditions.
    - Logs entry price, stop-loss, target, position size.
    - Implements a time-based exit if the price does not hit target or stop-loss.
    """
    trades = []
    trade_counts = {}  # Track the number of trades per day

    df['ExitDate'] = None  # Track when the trade exits

    for asset in df["Asset"].unique():
        asset_df = df[df["Asset"] == asset].copy()

        for i in range(len(asset_df)):
            row = asset_df.iloc[i]
            trade_date = row['Date']

            # Initialize trade count for the date if not already tracked
            if trade_date not in trade_counts:
                trade_counts[trade_date] = 0

            # Skip if the number of trades for this day exceeds the limit
            if trade_counts[trade_date] >= max_trades_per_day:
                continue  # Move to the next iteration

            if row['BullishBreakout'] or row['BearishBreakout']:
                # entry_price = row['Close']
                if i + 1 < len(asset_df):
                    next_row = asset_df.iloc[i + 1]
                    entry_price = next_row['Open']  # Use next day's open
                    entry_date = next_row['Date']  # Entry happens next day
                else:
                    entry_price = row['Close']  # ✅ Last row: use current close
                    entry_date = row['Date']  # Entry on the same day (final trade)

                stop_loss = row['StopLoss']
                target = row['Target']
                position_size = row['PositionSize']

                # Compute Risk-Reward Ratio
                risk = abs(entry_price - stop_loss)
                reward = abs(target - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0  # Avoid division by zero

                # Skip trades with poor RR ratio
                if rr_ratio < 2.0:
                    # print(f"Skipping trade for {asset} on {entry_date} (RR: {rr_ratio:.2f})")
                    continue  # Skip this trade

                # Look ahead up to 'exit_days' to check for stop-loss or target hit
                for j in range(i + 1, min(i + exit_days + 1, len(asset_df))):
                    future_row = asset_df.iloc[j]

                    # If price hits target
                    if (row['BullishBreakout'] and future_row['High'] >= target) or \
                       (row['BearishBreakout'] and future_row['Low'] <= target):
                        exit_date = future_row['Date']
                        exit_price = target
                        break

                    # If price hits stop-loss
                    elif (row['BullishBreakout'] and future_row['Low'] <= stop_loss) or \
                         (row['BearishBreakout'] and future_row['High'] >= stop_loss):
                        exit_date = future_row['Date']
                        exit_price = stop_loss
                        break

                    else:
                        # If neither stop-loss nor target is hit within exit_days, exit at last price
                        exit_date = asset_df.iloc[min(i + exit_days, len(asset_df) - 1)]['Date']
                        exit_price = asset_df.iloc[min(i + exit_days, len(asset_df) - 1)]['Close']
                   
                    # print(f"Bullish Breakouts: {df['BullishBreakout'].sum()}")
                    # print(f"Bearish Breakouts: {df['BearishBreakout'].sum()}")

                exit_price = exit_price * (1 - slippage_pct) if row['BullishBreakout'] else exit_price * (1 + slippage_pct)

                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'Asset': asset,
                    'Direction': 'Bullish' if row['BullishBreakout'] else 'Bearish',
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'StopLoss': stop_loss,
                    'Target': target,
                    'PositionSize': position_size,
                    'Contracts': row.get('Contracts', 1)  # Ensure 'Contracts' is included, default to 1 if missing
                })
                trade_counts[trade_date] += 1

    # Convert trades list to DataFrame before PnL calculations
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df = calculate_futures_pnl(trades_df)  # Apply PnL calculation only if trades exist

    return trades_df

# === Main Execution Function === #
def main():
    """
    Load market data, process indicators, identify breakouts, and execute trades.
    - Reads asset and market volume data from CSV.
    - Applies all calculation functions.
    - Simulates trades and generates output.
    - Plots trade results for visualization.
    """
    df = pd.read_csv('trade/asset_data.csv')  # Assume preprocessed CSV
    # Convert Date column to datetime and filter for 2024-2025 data
    # df["Date"] = pd.to_datetime(df["Date"])
    # df = df[df["Date"].dt.year >= 2023]

    # print(df.head())  # Debugging step: Check if data is correctly loaded
    
    df = calculate_indicators(df)
    df = integrate_volume_index(df)
    df = identify_breakouts(df)
    df = adjust_stop_loss_target(df)
    df = calculate_position_size(df)

        # Handle futures rollover
    df = handle_futures_rollover(df)  # <<< ADDED FUNCTION CALL HERE

    trades = simulate_trades(df)
    print(trades)
    trades.to_csv("trade/trades.csv", index=False)

if __name__ == '__main__':
    main()
    
