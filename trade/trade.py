import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Carver's risk multipliers
from multipliers import portfolio_risk_aggregator


# === Trading Logic Implementation === #
# This script assumes preprocessed data (Open, High, Low, Close, Volume) is provided from a CSV.
# It implements dynamic reward-to-risk, stop-loss adaptation, and volume-based position sizing.

def load_futures_specifications(csv_file="futures_specifications.csv"):
    """
    Loads futures contract specifications from a CSV file.
    """
    df = pd.read_csv(csv_file, parse_dates=["ExpiryDate"])  # Read CSV
    futures_dict = {}

    for _, row in df.iterrows():
        futures_dict[row["Ticker"]] = {
            "tick_size": row["TickSize"],
            "tick_value": row["TickValue"],
            "contract_size": row["ContractSize"],
            "margin_requirement": row["MarginRequirement"],
            "expiry_date": row["ExpiryDate"]
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
       
       
        # asset_df['ATR'] = asset_df['Range'].rolling(window=asset_df['Lookback']).mean()
        # asset_df['DRI'] = asset_df['Range'] / asset_df['Range'].rolling(window=asset_df['Lookback']).mean()
        # asset_df['AvgVolume'] = asset_df['Volume'].rolling(window=asset_df['Lookback']).mean()
                # Apply rolling window dynamically using Lookback column
        asset_df['ATR'] = asset_df.apply(lambda row: 
                                         asset_df.loc[:row.name, 'Range'].rolling(window=row['Lookback']).mean().iloc[-1], axis=1)

        asset_df['DRI'] = asset_df.apply(lambda row: 
                                         row['Range'] / asset_df.loc[:row.name, 'Range'].rolling(window=row['Lookback']).mean().iloc[-1], axis=1)

        asset_df['AvgVolume'] = asset_df.apply(lambda row: 
                                               asset_df.loc[:row.name, 'Volume'].rolling(window=row['Lookback']).mean().iloc[-1], axis=1)

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
        # Consolidation Detection
        asset_df['Consolidation'] = asset_df['DRI'].rolling(window=3, min_periods=1).apply(lambda x: all(x < 0.8), raw=True).astype(bool)

        # Consolidation High/Low using forward fill
        asset_df['ConsolidationHigh'] = asset_df['High'].where(asset_df['Consolidation']).ffill()
        asset_df['ConsolidationLow'] = asset_df['Low'].where(asset_df['Consolidation']).ffill()

        # Breakout Signals
        asset_df['BullishBreakout'] = (asset_df['Close'] > asset_df['ConsolidationHigh']) & (asset_df['VAS'] > 1.5)
        asset_df['BearishBreakout'] = (asset_df['Close'] < asset_df['ConsolidationLow']) & (asset_df['VAS'] > 1.5)
        
        all_data.append(asset_df)

        print(asset_df[['Close', 'ConsolidationHigh', 'ConsolidationLow', 'VAS']].dropna().head())

    return pd.concat(all_data, ignore_index=True)

# === Function to Integrate Volume Index Comparison === #
def integrate_volume_index(df):
    """
    Compare asset volume to a broader market volume benchmark.
    - Ensures each asset is compared to its corresponding market sector.
    """
    # df = df.merge(market_df, on=["Date", "Sector"], how="left")
    # df['MarketAvgVolume'] = df['MarketVolume'].rolling(window=df['Lookback']).mean()
    
        # Ensure `Sector` is assigned to df from market_df
    # Ensure 'Sector' column exists in `df`
    # if 'Sector' not in df.columns:        
    #     # Extract unique sector mappings from market_df (assumes market_df has one sector per asset)
    #     sector_mapping = market_df[['Sector']].drop_duplicates()

    #     # Merge `df` with `market_df` to get `Sector`
    #     df = df.merge(sector_mapping, on="Sector", how="left")

    # # Now merge with market volume data
    # df = df.merge(market_df[['Date', 'Sector', 'MarketVolume']], on=["Date", "Sector"], how="left")

    # # Compute Market Volume Anomaly Score (MarketVAS)
    # df['MarketAvgVolume'] = df.groupby('Sector')['MarketVolume'].transform(lambda x: x.rolling(window=df['Lookback'].iloc[0]).mean())
   
    # Compute Market Volume Anomaly Score (MarketVAS)
    df['MarketAvgVolume'] = df.groupby('Sector')['SectorMarketVolume'].transform(lambda x: x.rolling(window=df['Lookback'].iloc[0]).mean())
    df['MarketVAS'] = df['SectorMarketVolume'] / df['MarketAvgVolume']

    # df['MarketVAS'] = df['MarketVolume'] / df['MarketAvgVolume']
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
        # asset_df['StopLoss'] = np.where(asset_df['BullishBreakout'], 
        #                                 asset_df['Close'] - (0.3 * asset_df['ATR']), 
        #                                 asset_df['Close'] + (0.3 * asset_df['ATR']))
        
                # Reduce stop-loss range dynamically if price is consolidating
        asset_df['ATR_Stop_Adjust'] = np.where(asset_df['Consolidation'], 0.5 * asset_df['ATR'], asset_df['ATR'])
        asset_df['StopLoss'] = np.where(asset_df['BullishBreakout'],
                                        asset_df['Close'] - (0.3 * asset_df['ATR_Stop_Adjust']),
                                        asset_df['Close'] + (0.3 * asset_df['ATR_Stop_Adjust']))

        asset_df['RewardToRisk'] = (asset_df['ATR'] / (asset_df['ConsolidationHigh'] - asset_df['ConsolidationLow'])).clip(1.5, 3.0)
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
    Rolls over expiring futures contracts into the next available month.
    """
    for asset in FUTURES_SPECIFICATIONS:
        expiry_date = FUTURES_SPECIFICATIONS[asset].get('expiry_date')

        if expiry_date and pd.Timestamp.today() >= expiry_date:
            print(f"Rollover: Closing {asset} and rolling into next contract month.")
            df.loc[df['Asset'] == asset, 'Rollover'] = True  # Flag trades for rollover
    
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

    print("Debug - Checking Contracts Column:")
    print(trades[['Asset', 'Contracts']].dropna().head())  # Ensure Contracts exist
    Grss_pnl=0
    Nt_pnl=0
    for i, trade in trades.iterrows():
        asset = trade['Asset']
        if asset in FUTURES_SPECIFICATIONS:
            spec = FUTURES_SPECIFICATIONS[asset]
            tick_size, tick_value, contract_size = spec['tick_size'], spec['tick_value'], spec['contract_size']

            # Compute P&L using futures-specific calculations
            price_difference = trade['ExitPrice'] - trade['EntryPrice']
            tick_movement = price_difference / tick_size  # Number of ticks moved
            gross_pnl = tick_movement * tick_value * contract_size * trade['Contracts']

            # Apply transaction costs (per contract per trade)
            # total_transaction_cost = trade['Contracts'] * spec['margin_requirement'] * 0.0001  # Example: 0.01% cost
            total_transaction_cost = calculate_futures_transaction_cost(trade)  # <<< ADDED FUNCTION CALL HERE

            # Compute net P&L
            net_pnl = gross_pnl - total_transaction_cost

            # Update trade DataFrame
            trades.at[i, 'GrossPnL'] = gross_pnl
            Grss_pnl+=gross_pnl
            trades.at[i, 'TransactionCost'] = total_transaction_cost
            trades.at[i, 'NetPnL'] = net_pnl
            Nt_pnl+=net_pnl
    print("grosspnl",Grss_pnl)
    print("ntpnl",Nt_pnl)

    return trades

# === Function to Calculate Dynamic Position Sizing === #
def calculate_position_size(df, capital=10000000, risk_per_trade=0.01, atr_multiplier=1.5, max_leverage=3.0, max_correlation_risk=0.2, max_portfolio_volatility=0.15, max_jump_risk=0.1):
    """
    Determines position size per asset using both:
    1. ATR-based volatility scaling (reduces size in high-volatility markets)
    2. VAS-based volume scaling (prioritizes high-confidence setups)

    - Ensures dynamic risk-adjusted position sizing.
    - Balances volatility risk and volume-based opportunities.
    """
    assets = df["Asset"].unique()
    all_data = []
    positions = []

    for asset in assets:
        asset_df = df[df["Asset"] == asset].copy()
        risk_amount = capital * risk_per_trade  # Risk allocated per trade
        
        # Volatility-adjusted risk (higher ATR = smaller position size)
        asset_df['ATR_Adjusted_Size'] = risk_amount / (asset_df['ATR'] * atr_multiplier)

        # Apply volume-based scaling (VAS ensures high-confidence setups get more allocation)
        asset_df['PositionSize'] = asset_df['ATR_Adjusted_Size'] * np.minimum(asset_df['VAS'], 3)

        # FUTURES-SPECIFIC POSITION SIZING
        if asset in FUTURES_SPECIFICATIONS:
            contract_spec = FUTURES_SPECIFICATIONS[asset]
            contract_size = contract_spec['contract_size']
            tick_value = contract_spec['tick_value']
            margin_requirement = contract_spec['margin_requirement']

            print(f"Debug - PositionSize for {asset}:")
            print(asset_df[['Date', 'Asset', 'PositionSize']].dropna().head())  # Show first few rows
            asset_df['PositionSize'] = asset_df['PositionSize'].fillna(0)

            print(f"Debug - Total Capital Used for {asset}: {asset_df['PositionSize'].sum()} / {capital}")

            # # Ensure capital is used efficiently without overwriting ATR/VAS-based PositionSize
            # total_allocated = df.groupby('Asset')['PositionSize'].sum()
            # df['CapitalShare'] = df['PositionSize'] / total_allocated  
            # df['PositionSize'] = capital * df['CapitalShare']

            # # Debug normalized PositionSize
            # print("Debug - Normalized PositionSize Distribution:", df.groupby('Asset')['PositionSize'].sum())

            # Compute Contracts
            contract_value = contract_size * tick_value
            print(contract_value)
            asset_df['Contracts'] = asset_df['PositionSize'] / contract_value

            # Apply margin constraint and rounding in one step
            max_contracts = capital / margin_requirement
            print(max_contracts)
            asset_df['Contracts'] = np.minimum(np.ceil(asset_df['Contracts']), max_contracts).astype(int)

            # Debugging print
            print(f"Debug - Assigned Contracts for {asset}: {asset_df['Contracts'].unique()}")

        # Store positions for risk aggregation
        positions.append(asset_df['Contracts'].values)

        all_data.append(asset_df)

    df = pd.concat(all_data, ignore_index=True)
    covariance_matrix, jump_covariance_matrix = calculate_risk_matrices(df)

    # Convert positions to NumPy array
    positions = np.array(positions)

    # Compute positions weighted per asset
    unique_assets = df["Asset"].unique()
    positions_weighted = np.zeros(len(unique_assets))

    for i, asset in enumerate(unique_assets):
        # Extract positions for the current asset across all dates
        asset_positions = positions[:, i]  
        mean_position = np.mean(asset_positions)
        positions_weighted[i] = mean_position / capital

    positions_weighted = np.array(positions_weighted)

    # Apply Carver's Risk Multipliers
    final_positions = portfolio_risk_aggregator(
        positions=positions,
        positions_weighted=np.array(positions_weighted).flatten(),
        covariance_matrix=covariance_matrix.to_numpy(),
        jump_covariance_matrix=jump_covariance_matrix.to_numpy(),
        maximum_portfolio_leverage=max_leverage,
        maximum_correlation_risk=max_correlation_risk,
        maximum_portfolio_risk=max_portfolio_volatility,
        maximum_jump_risk=max_jump_risk,
        date=pd.Timestamp.today()
    )

    # Update df with adjusted position sizes
    df['PositionSize'] = final_positions.flatten()
    df['Contracts'] = final_positions.flatten()

    return df

# === Function to Simulate Trades === #
def simulate_trades(df, exit_days=5, transaction_cost_pct=0.001, slippage_pct=0.0005):
    """
    Execute trade simulation per asset.
    - Identifies trade entries based on breakout conditions.
    - Logs entry price, stop-loss, target, position size.
    - Implements a time-based exit if the price does not hit target or stop-loss.
    """
    trades = []

    # trades = pd.DataFrame(trades)  # Convert to DataFrame before passing
    # trades = calculate_futures_pnl(trades)  # Apply PnL calculation to all trades

    df['ExitDate'] = None  # Track when the trade exits

    for asset in df["Asset"].unique():
        asset_df = df[df["Asset"] == asset].copy()

        for i in range(len(asset_df)):
            row = asset_df.iloc[i]
            
            if row['BullishBreakout'] or row['BearishBreakout']:
                entry_price = row['Close']
                stop_loss = row['StopLoss']
                target = row['Target']
                position_size = row['PositionSize']
                entry_date = row['Date']

                # Compute Risk-Reward Ratio
                risk = abs(entry_price - stop_loss)
                reward = abs(target - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0  # Avoid division by zero

                # Skip trades with poor RR ratio
                if rr_ratio < 1.5:
                    print(f"Skipping trade for {asset} on {entry_date} (RR: {rr_ratio:.2f})")
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
                   
                    print(f"Bullish Breakouts: {df['BullishBreakout'].sum()}")
                    print(f"Bearish Breakouts: {df['BearishBreakout'].sum()}")

                # Apply slippage to exit price
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
    df = pd.read_csv('asset_data.csv')  # Assume preprocessed CSV
    # market_df = pd.read_csv('market_volume.csv')  # Broader market volume benchmark
    
    print(df.head())  # Debugging step: Check if data is correctly loaded
    
    df = calculate_indicators(df)
    df = identify_breakouts(df)
    df = integrate_volume_index(df)
    df = adjust_stop_loss_target(df)
    df = calculate_position_size(df)

        # Handle futures rollover
    df = handle_futures_rollover(df)  # <<< ADDED FUNCTION CALL HERE

    trades = simulate_trades(df)
    print(trades)
    trades.to_csv("trades.csv", index=False)

if __name__ == '__main__':
    main()
    
