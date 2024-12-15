import pandas as pd

def compute_headline_length(df):
    """Add a column with headline lengths."""
    df['headline_length'] = df['headline'].apply(len)
    return df

def compute_basic_statistics(df):
    """Compute and return basic statistics."""
    return df.describe()

def find_outliers(df, column):
    """Identify outliers based on 2 standard deviations."""
    mean = df[column].mean()
    std_dev = df[column].std()
    high_threshold = mean + 2 * std_dev
    low_threshold = mean - 2 * std_dev
    return df[(df[column] > high_threshold) | (df[column] < low_threshold)]
