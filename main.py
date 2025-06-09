from data_loader import load_train_data
from feature_engineering import preprocess_train_data
from model import train_model, evaluate_model
from visualize import plot_all_features


def main():
    data = load_train_data()
    X, y = preprocess_train_data(data)

    model = train_model(X, y)
    evaluate_model(model, X, y)

    plot_all_features(X, y)

if __name__ == "__main__":
    main()
