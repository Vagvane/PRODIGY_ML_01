import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse}")
    # Print one sample prediction
    sample_input = [[1500, 3, 2]]  # Example: 1500 sqft, 3 bedrooms, 2 baths
    sample_price = model.predict(sample_input)[0]
    print(f"Predicted price for house with 1500 sqft, 3 beds, 2 baths: ${sample_price:,.2f}")
    return mse
