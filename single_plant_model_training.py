import numpy as np
from sklearn.ensemble import IsolationForest


class ExpertEnsemble:
    def __init__(self, contamination='auto'):
        self.contamination = contamination
        self.experts = {}
        self.plant_registry = []

    def fit(self, X_features, plant_ids, true_labels):
        self.plant_registry = np.unique(plant_ids)
        print(f"Training {len(self.plant_registry)} localized single plant models (Healthy Unilateral Baseline Only).")

        for plant in self.plant_registry:
            subject_mask = (plant_ids == plant) & (true_labels == 0)
            X_subject = X_features[subject_mask]

            if len(X_subject) == 0:
                print(f"WARNING: Plant {plant} contains zero healthy unilateral baseline data. Skipping.")
                continue

            model = IsolationForest(n_estimators=100, contamination=self.contamination, random_state=42)
            model.fit(X_subject)
            self.experts[plant] = model

    def transform(self, X_features):
        print("Generating unilateral anomaly score matrix for each plant model.")
        meta_features = []
        for plant in self.plant_registry:
            scores = self.experts[plant].decision_function(X_features)
            meta_features.append(scores)
        return np.column_stack(meta_features)