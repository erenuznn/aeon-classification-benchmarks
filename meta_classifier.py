from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


class SubjectMetaClassifier:
    def __init__(self, n_estimators=200, max_depth=10):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    def fit(self, meta_features, labels):
        print("Training Meta-Classifier on Anomaly Matrix.")
        self.classifier.fit(meta_features, labels)

    def predict_and_evaluate(self, meta_features, true_labels):
        print("Executing Meta-Classifier inference.")
        predictions = self.classifier.predict(meta_features)

        acc = accuracy_score(true_labels, predictions)
        print(f"PIPELINE ACCURACY: {acc:.4f}")
        print("CLASSIFICATION REPORT:")
        print(classification_report(true_labels, predictions))
        return predictions

    def predict_probabilities(self, meta_features):
        """
        Extracts the raw probability [0.0 to 1.0] of the 'Stressed' class (1).
        """
        print("Extracting chronological stress probabilities.")
        classes = self.classifier.classes_

        # Ensure the stressed class exists in the training data mapping
        if 1 in classes:
            stress_index = np.where(classes == 1)[0][0]
            return self.classifier.predict_proba(meta_features)[:, stress_index]
        else:
            print("WARNING: Class '1' not found in model. Returning zeros.")
            return np.zeros(len(meta_features))