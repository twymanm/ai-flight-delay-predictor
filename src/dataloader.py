import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV Data from the given path
    """
    return pd.read_csv(path, parse_dates=["FL_DATE"])

def prepare_ontime_data(df: pd.DataFrame):
    """
    Clean and preprocess raw on-time flight data for machine learning.
    Returns: X_train, X_test, y_train, y_test
    """

    # Create a binary target variable: 1 if arrival delay >= 15 minutes, else 0
    df["Delayed"] = df["ARR_DELAY_NEW"].fillna(0).apply(lambda x: 1 if x >= 15 else 0)

    # Drop rows where key features are missing
    df.dropna(subset=["ORIGIN", "DEST", "OP_UNIQUE_CARRIER"], inplace=True)

    # Extract month and day of week from the flight date
    df["Month"] = df["FL_DATE"].dt.month
    df["DayOfWeek"] = df["FL_DATE"].dt.dayofweek  # Monday=0, Sunday=6

    # Select features for training
    X = df[["Month", "DayOfWeek", "ORIGIN", "DEST", "OP_UNIQUE_CARRIER"]]

    # Target variable: Delayed (binary classification)
    y = df["Delayed"]

    # One-hot encode categorical features (airports and carrier)
    X_encoded = pd.get_dummies(X, columns=["ORIGIN", "DEST", "OP_UNIQUE_CARRIER"])

    # Split the data into training and test sets (80/20 split, stratified by target)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


