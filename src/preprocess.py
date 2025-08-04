import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.drop(columns=["Delayed"])
    y = df["Delayed"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=["Origin", "Destination", "Airline"])

    return X_encoded, y