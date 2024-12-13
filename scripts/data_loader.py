#data_loader.py

import pandas as pd
import os

def load_data(file_path):
    """
    Loads a CSV dataset from the specified file path.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    # Load dataset
    df = pd.read_csv(file_path)
    
    # Preview the dataset
    print("Dataset Loaded Successfully!")
    print("Columns in Dataset:", df.columns)
    print("Data Types:")
    print(df.dtypes)
    
    return df
