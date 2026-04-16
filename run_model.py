import joblib
from aeon.datasets import load_from_ts_file

# 1. Absolute Path Definition
MODEL_PATH = "/Users/erenuzun/Desktop/Thesis/ML/MODELS/rocket_unilateral_v1.joblib"
# Assuming new data is stored in the standard unilateral data directory
DATA_PATH = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts/new_plant_data_unilateral.ts"

# 2. Load the Unilateral Model
model = joblib.load(MODEL_PATH)

# 3. Load New Unilateral Data
X_new, _ = load_from_ts_file(DATA_PATH)

# 4. Execute Predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Output Results
print(f"Prediction for first instance: {predictions[0]}")
print(f"Probability of class 1: {probabilities[0, 1]:.2f}")