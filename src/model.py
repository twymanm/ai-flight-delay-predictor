from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y):
    """
    Train a random forest on the data and return the trained model
    """
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X, y)
    preds = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, preds):.2f}")
    return model
