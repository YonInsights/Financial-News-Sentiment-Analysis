import pandas as pd
import os

def load_data(file_path):
    """
    Loads and preprocesses a CSV dataset from the specified file path.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded and preprocessed dataset.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S%z", errors='coerce')
    
    # Extract useful features
    df['date_only'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Preview the dataset
    print("Dataset Loaded and Preprocessed Successfully!")
    print("Columns in Dataset:", df.columns)
    print("Data Types:")
    print(df.dtypes)
    
    return df
