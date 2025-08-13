from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
   # Choose ONE of these paths:
    #path = "data/sample.csv"                # small toy data
    path = "data/T_ONTIME_REPORTING.csv"  # real BTS data

    X, y = preprocess_data(path)

    # Use a split only if you have enough rows (BTS file). For tiny sample.csv, skip split.
    test_size = 0.2 if len(X) > 50 else 0.0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X.iloc[:0], y.iloc[:0]  # empty test for tiny datasets

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        print("Accuracy:", f"{accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred, digits=3))
    else:
        # For tiny datasets, just show training accuracy
        print("Training accuracy (tiny dataset):",
              f"{accuracy_score(y_train, model.predict(X_train)):.3f}")


if __name__ == "__main__":
    main()