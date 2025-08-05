import pandas as pd

def preprocess_data(filepath):
    """
    Load and preprocess flight data from CSV.
    - Reads the CSV file
    - Splts features and target label
    - One-hot encodes categorical columns

    Parameters:
    filepath (str): Path to the input CSV file

    Returns:
    X_encoded (DataFrame): Preprocessed feature matrix
    y (Series): Target labels (0 or 1 for on-time vs delayed)
    """
    #load the CSV
    df = pd.read_csv(filepath)

    #Ensure correct column names (case-sensitive)
    #Optional: rename to standardize

    df = df.rename(columns={
        "ORIGIN": "Origin",
        "DEST": "Destination",
        "OP_UNIQUE_CARRIER": "Airline"
    })

    #Create binary label if not already there (optional safety)
    if "Delayed" not in df.columns:
        df["Delayed"] = df["ARR_DELAY_NEW"].fillna(0).apply(lambda x: 1 if x >= 15 else 0)

    #Drop missing values in key columns
    df.dropna(subset=["Origin", "Destination", "Airline"], inplace = True)

    #select features and lavel
    X = df[["Origin", "Destination", "Airline", "FL_DATE"]].copy()
    y = df["Delayed"]

    #Extract time features
    X["FL_DATE"] = pd.to_datetime(X["FL_DATE"])
    X["Month"] = X["FL_DATE"].dt.month
    X["DayOfWeek"] = X["FL_DATE"].dt.dayofweek
    X.drop(columns=["FL_DATE"], inplace=True)

    #One hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=["Origin", "Destination", "Airline"])

    return X_encoded, y


    
    
    # Separate features and labels
    X = df.drop(columns=["Delayed"])
    y = df["Delayed"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=["Origin", "Destination", "Airline"])

    return X_encoded, y