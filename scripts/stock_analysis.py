import pandas as pd

def calculate_daily_returns(df, date_column, price_column):
    """
    Calculate daily stock returns as percentage change.
    Args:
        df (pd.DataFrame): Stock price data.
        date_column (str): Name of the date column.
        price_column (str): Name of the closing price column.
    Returns:
        pd.DataFrame: Dataframe with daily returns added.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)
    df['daily_return'] = df[price_column].pct_change()
    return df
