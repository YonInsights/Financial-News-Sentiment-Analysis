from scipy.stats import pearsonr

def merge_sentiment_and_returns(sentiment_df, stock_df):
    """
    Merge daily sentiment scores with stock returns on date.
    Args:
        sentiment_df (pd.DataFrame): Daily aggregated sentiment scores.
        stock_df (pd.DataFrame): Daily stock returns.
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    return pd.merge(sentiment_df, stock_df, left_on='date', right_on='Date')

def calculate_correlation(merged_df, sentiment_column='average_sentiment', return_column='daily_return'):
    """
    Calculate Pearson correlation between sentiment scores and stock returns.
    Args:
        merged_df (pd.DataFrame): Dataframe with sentiment and returns data.
    Returns:
        float: Pearson correlation coefficient.
    """
    correlation, _ = pearsonr(merged_df[sentiment_column], merged_df[return_column])
    return correlation
