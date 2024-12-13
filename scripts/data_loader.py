import pandas as pd
import os

def load_data(file_path):
    """Load CSV data from the specified file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    return pd.read_csv(file_path)
