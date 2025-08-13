import pandas as pd

def preprocess_data(filepath: str):
    """
    Load and preprocess either:
      (A) sample.csv schema: FlightNumber,Origin,Destination,Distance,Airline,Delayed
      (B) BTS on-time schema: ... FL_DATE, OP_UNIQUE_CARRIER, ORIGIN, DEST, ARR_DELAY_NEW ...
    Returns:
      X_encoded (DataFrame), y (Series)
    """
    df = pd.read_csv(filepath)

    # ---------- Case A: simple sample.csv ----------
    if {"Origin", "Destination", "Airline", "Delayed"}.issubset(df.columns):
        # Minimal features + one-hot for categoricals
        # Distance is numeric already; encode the categorical fields
        X = df[["Origin", "Destination", "Airline", "Distance"]].copy()
        y = df["Delayed"].astype(int)
        X_encoded = pd.get_dummies(X, columns=["Origin", "Destination", "Airline"])
        return X_encoded, y

    # ---------- Case B: BTS on-time schema ----------
    # Normalize some column names we rely on
    rename_map = {
        "ORIGIN": "Origin",
        "DEST": "Destination",
        "OP_UNIQUE_CARRIER": "Airline",
        "DAY_OF_WEEK": "DayOfWeek",
        "MONTH": "Month",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Create Delayed label if needed
    if "Delayed" not in df.columns:
        if "ARR_DELAY_NEW" in df.columns:
            df["Delayed"] = df["ARR_DELAY_NEW"].fillna(0).apply(lambda x: 1 if x >= 15 else 0)
        elif "ARR_DEL15" in df.columns:
            df["Delayed"] = df["ARR_DEL15"].fillna(0).astype(int)
        else:
            raise ValueError("Cannot find ARR_DELAY_NEW or ARR_DEL15 to derive Delayed label.")

    # Parse date if present (for Month/DayOfWeek if originals missing)
    if "FL_DATE" in df.columns and pd.api.types.is_string_dtype(df["FL_DATE"]):
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

    if "Month" not in df.columns:
        if "FL_DATE" in df.columns:
            df["Month"] = df["FL_DATE"].dt.month
        elif "MONTH" in df.columns:
            df["Month"] = df["MONTH"]
        else:
            df["Month"] = pd.NA

    if "DayOfWeek" not in df.columns:
        if "FL_DATE" in df.columns:
            # Monday=0..Sunday=6
            df["DayOfWeek"] = df["FL_DATE"].dt.dayofweek
        elif "DAY_OF_WEEK" in df.columns:
            # BTS uses 1..7; shift to 0..6
            df["DayOfWeek"] = df["DAY_OF_WEEK"].astype(int) - 1
        else:
            df["DayOfWeek"] = pd.NA

    # Keep only rows with essential fields
    needed = ["Origin", "Destination", "Airline", "Month", "DayOfWeek", "Delayed"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    # Build feature matrix
    feat_cols = ["Origin", "Destination", "Airline", "Month", "DayOfWeek"]
    X = df[feat_cols].copy()
    y = df["Delayed"].astype(int)

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, columns=["Origin", "Destination", "Airline"], drop_first=False)
    return X_encoded, y