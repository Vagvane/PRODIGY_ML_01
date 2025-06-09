import pickle
import pandas as pd
from data_loader import load_test_data
from feature_engineering import preprocess_test_data

def test_pipeline():
    with open("linear_model.pkl", "rb") as f:
        model = pickle.load(f)

    test_df = load_test_data()
    X_test = preprocess_test_data(test_df)
    predictions = model.predict(X_test)

    output = pd.DataFrame({
        'Id': test_df.get('Id', pd.Series(range(1, len(predictions) + 1))),
        'SalePrice': predictions
    })
    output.to_csv("submission.csv", index=False)
    print("\u2705 Predictions saved to submission.csv")
    print("\nSample Predictions:")
    print(output.head())

if __name__ == "__main__":
    test_pipeline()