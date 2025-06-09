from data_loader import load_data
from feature_engineering import prepare_features
from model import train_model
from predict import predict
from visualize import plot_feature_vs_target


def main():
    # Load the data
    df = load_data()

    # Prepare the features and target variable
    X, y = prepare_features(df)

    plot_feature_vs_target(X,y,"GrLivArea")
    plot_feature_vs_target(X, y, "BedroomAbvGr")
    plot_feature_vs_target(X, y, "FullBath")
    # plot_correlation_heatmap(df[['GrLivArea','BedroonAbvGr','FullBath','SalePrice']])

    # Train the model
    model= train_model(X, y)

    # Example sample data for prediction
    sample_house = [1500, 3, 2]  # Example: GrLivArea=1500, BedroomAbvGr=3, FullBath=2

    # Make a prediction
    price = predict(model, sample_house)
    print(f"Predicted price for the house:${price:.2f}")

if __name__=="__main__":
    main()