import numpy as np
from aeon.classification.convolution_based import RocketClassifier

# ==========================================
# --- DATA INITIALIZATION ---
# ==========================================
# Generation of a synthetic 3D array representing time-series data
# Dimensions: (n_instances, n_channels, n_timepoints)
X_train = np.random.normal(0, 1, (10, 1, 50))
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
X_test = np.random.normal(0, 1, (2, 1, 50))

print("Function: Executes unilateral classification sequence. Combines the Rocket transformer with a Ridge Classifier.")

rocket_model = RocketClassifier(n_kernels=1000)

# Method: fit()
# Ingests the 3D array and 1D label array to train the model.
rocket_model.fit(X_train, y_train)

# Method: predict()
# Ingests a new 3D array and outputs predicted discrete class labels.
predictions = rocket_model.predict(X_test)

# Method: predict_proba()
# Outputs the probability estimates for each class.
probabilities = rocket_model.predict_proba(X_test)