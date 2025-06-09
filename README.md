# House Price Prediction 🏠

This project predicts house prices using a Linear Regression model based on features like square footage, number of bedrooms, and bathrooms.

## 🔧 Technologies Used
- Python 3.13
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 📁 Project Structure
- `data_loader.py` – Loads the dataset
- `feature_engineering.py` – Prepares features and target
- `model.py` – Trains the model
- `predict.py` – Predicts using the model
- `visualize.py` – Plots relationships
- `main.py` – Main controller file
- `requirements.txt` – Package list

## 🚀 How to Run
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
