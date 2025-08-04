import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV Data from the given path
    """
    return pd.read_csv(path)