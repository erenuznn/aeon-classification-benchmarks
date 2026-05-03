from aeon.classification.convolution_based import RocketClassifier

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




