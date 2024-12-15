import pynance as pn

def calculate_metrics(df):
    """Calculate daily and cumulative returns."""
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
    return df
