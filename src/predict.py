from preprocess import preprocess_data
from model import train_model

def main():
    X, y = preprocess_data("data/sample.csv")
    model = train_model(X, y)

if __name__ == "__main__":
    main()