import pandas as pd
import ta

def add_indicators(df):
    df = df.copy()  
    
    # Simple Moving Average
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)

    # Momentum Indicators
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])  #12 day EMA - 26 day EMA

    # Volatility Indicators
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'], window=20) #20 day SMA + 2 SD
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20) #20 day SMA - 2 SD
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14) # 14 day ATR using SMA

    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['CMF_20'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    df['Vol_20'] = ta.trend.sma_indicator(df['Volume'], window=20) #20 day Volume SMA

    # Our Indicators
    df['DRI'] = df['Close']/df['SMA_20']
    df['VAS'] = df['Volume']/df['Vol_20']
    return df
    
df = pd.read_csv('ohlcv_1d.csv')
df = df.rename(columns = {'time': 'Time', 'symbol':'Symbol', 'open':'Open','high':'High','low': 'Low', 'close': 'Close', 'volume': 'Volume'}) #function uses capitalized titles, whereas our database uses lowercase

futures_dict = {Symbol: add_indicators(df[df['Symbol'] == Symbol]) for Symbol in df['Symbol'].unique()} # Applies all the indicators to dataset sequentially for all unique symbols (assets)
indicators_df = pd.concat(futures_dict.values(), ignore_index=True)
indicators_df = indicators_df.sort_values(by = ['Time','Symbol'], ascending = [False,True]) #our SQL database has the earliest dates on top
indicators_df.to_csv("indicators_and_data.csv", index=False)

metadf = pd.read_csv("metadata.csv")
metadf = metadf.rename(columns={'Databento Symbol' : 'Symbol'})
metadf['Symbol'] = metadf['Symbol'] + '.c.0'

def addmetadata(df):
    return df.merge(metadf, on="Symbol", how="left")

final = addmetadata(indicators_df)

# CSV 1 for Trade Logic
market_volume = final[['Time','Symbol','Volume', 'VAS','Sector']]
market_volume = market_volume.rename(columns = {'Time':'Date','Volume':'MarketVolume'})
market_volume.to_csv("market_volume.csv", index = False)

#CSV 2
almost_asset = final
almost_asset = almost_asset.copy()
almost_asset['Lookback'] = 20
almost_asset = almost_asset[['Time','Symbol','Open','High','Low','Close','Volume','Lookback']]
almost_asset['SectorMarketVolume'] = 0
almost_asset = almost_asset.rename(columns = {'Time': 'Date', 'Symbol':'Asset'})
almost_asset['Date'] = pd.to_datetime(almost_asset['Date']).dt.strftime('%Y-%m-%d')
almost_asset['Asset'] = almost_asset['Asset'].str.replace('.c.0', '', regex=False)
almost_asset = almost_asset.sort_values(by = ['Asset','Date'], ascending= [True, True])
almost_asset.head(5)
almost_asset.to_csv('asset_data.csv', index=False)
