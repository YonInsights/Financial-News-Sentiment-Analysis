import pandas as pd
import os

def load_data(file_path: str, required_columns: list = None) -> pd.DataFrame:
    """
    Loads and preprocesses a CSV dataset from the specified file path.
    Performs validation and basic preprocessing, including renaming and
    normalizing the date format.

    Args:
        file_path (str): Path to the CSV file.
        required_columns (list): List of required columns to check.

    Returns:
        pd.DataFrame: The loaded and preprocessed dataset.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Validate required columns
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

    # Normalize column names and date format
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d", errors='coerce')

        # Drop invalid date values
        initial_length = len(df)
        df.dropna(subset=['date'], inplace=True)
        final_length = len(df)
        if initial_length != final_length:
            print(f"Dropped {initial_length - final_length} rows due to invalid date values.")
        
        # Add auxiliary date-related features for extended use cases
        df['date_only'] = df['date'].dt.date
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()

    # Preview the dataset for sanity check
    print("Dataset Loaded and Preprocessed Successfully!")
    print("Columns in Dataset:", df.columns)
    print("Data Types:")
    print(df.dtypes)

    return df

def prepare_news_data(file_path: str) -> pd.DataFrame:
    """
    Prepares the news dataset for further analysis.
    Ensures normalized date format and additional validation.

    Args:
        file_path (str): Path to the news dataset.

    Returns:
        pd.DataFrame: Preprocessed news dataset.
    """
    required_columns = ['date', 'headline']
    news_df = load_data(file_path)
    # Ensure only required columns are selected for news data
    return news_df[['date', 'headline']]

def prepare_stock_data(file_path: str) -> pd.DataFrame:
    """
    Prepares the stock dataset for further analysis.
    Ensures normalized date format and basic stock-related preprocessing.

    Args:
        file_path (str): Path to the stock dataset.

    Returns:
        pd.DataFrame: Preprocessed stock dataset.
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stock_df = load_data(file_path, required_columns)
    # Retain only relevant columns for stock analysis
    return stock_df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]

if __name__ == "__main__":
    # Example file paths (replace these with actual paths during testing)
    news_file = "data/news.csv"  # Replace with your news dataset path
    stock_file = "data/stock_prices.csv"  # Replace with your stock dataset path

    # Prepare the datasets
    news_data = prepare_news_data(news_file)
    stock_data = prepare_stock_data(stock_file)

    # Print previews of prepared datasets
    print(news_data.head())
    print(stock_data.head())
