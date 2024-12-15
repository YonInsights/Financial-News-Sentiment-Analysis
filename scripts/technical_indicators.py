import talib

def calculate_sma(df, period=20):
    """Calculate Simple Moving Average."""
    df['SMA'] = talib.SMA(df['Close'], timeperiod=period)
    return df

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    df['RSI'] = talib.RSI(df['Close'], timeperiod=period)
    return df

def calculate_macd(df):
    """Calculate MACD and Signal Line."""
    df['MACD'], df['Signal'], df['Hist'] = talib.MACD(df['Close'], 
                                                      fastperiod=12, 
                                                      slowperiod=26, 
                                                      signalperiod=9)
    return df
