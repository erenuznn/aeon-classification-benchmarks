import joblib
import pandas as pd
import numpy as np
from data_loader import load_unilateral_data
from feature_extractor import UnilateralFeatureExtractor
from single_plant_model_training import ExpertEnsemble
from meta_classifier import SubjectMetaClassifier
import plotly.graph_objects as go

# Configuration variables
BASE_DIRECTORY = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
LABEL_CSV_FILE = "/Users/erenuzun/Desktop/Thesis/ML/DATA/labels.csv"
STRESS_ONSET_DATE = "2025-08-22"
TRAIN_PLANT_IDS = ["25072221-1", "25072233-1", "25072236-1", "25072240-1", "25072249-1", "25072269-1", "25072279-1", "25072280-1", "25072283-1", "25072288-1", "25072290-1", "25072294-1", "25072300-1", "25072361-1"]
TEST_PLANT_IDS = ["25072219-1","25072205-1"]

print("Unilateral pipeline execution sequence initiated.")

# Data loading
X_train, time_train, id_train = load_unilateral_data(base_directory=BASE_DIRECTORY, plant_id_list=TRAIN_PLANT_IDS)
X_test, time_test, id_test = load_unilateral_data(base_directory=BASE_DIRECTORY, plant_id_list=TEST_PLANT_IDS)

# Chronological Label Mapping
labels_df = pd.read_csv(LABEL_CSV_FILE)
label_dict = dict(zip(labels_df['plant_id'], labels_df['label']))
stress_threshold = pd.to_datetime(STRESS_ONSET_DATE)

y_train = np.zeros(len(id_train), dtype=int)
for i in range(len(id_train)):
    destiny = label_dict.get(id_train[i], -1)
    if destiny == -1:
        y_train[i] = -1
    elif destiny == 1 and time_train.iloc[i] >= stress_threshold:
        y_train[i] = 1
    else:
        y_train[i] = 0

y_test = np.zeros(len(id_test), dtype=int)
for i in range(len(id_test)):
    destiny = label_dict.get(id_test[i], -1)
    if destiny == -1:
        y_test[i] = -1
    elif destiny == 1 and time_test.iloc[i] >= stress_threshold:
        y_test[i] = 1
    else:
        y_test[i] = 0

if -1 in y_train or -1 in y_test:
    print("WARNING: Unmapped unilateral labels detected. Verify CSV configuration.")

# Feature Extraction
extractor = UnilateralFeatureExtractor()
X_train_fused = extractor.fit_transform(X_train, time_train)
X_test_fused = extractor.transform(X_test, time_test)

# Expert Layer
expert_layer = ExpertEnsemble()
expert_layer.fit(X_train_fused, id_train, y_train)

meta_train = expert_layer.transform(X_train_fused)
meta_test = expert_layer.transform(X_test_fused)

# Meta-Classifier
print("Training and evaluating meta-classifier.")
meta_model = SubjectMetaClassifier()
meta_model.fit(meta_train, y_train)
meta_model.predict_and_evaluate(meta_test, y_test)

print("Rendering chronological probability timeline.")
stress_probs = meta_model.predict_probabilities(meta_test)

fig = go.Figure()

# Plot a separate line for each test subject
for plant in np.unique(id_test):
    plant_mask = (id_test == plant)

    fig.add_trace(go.Scattergl(
        x=time_test.iloc[plant_mask] if isinstance(time_test, pd.Series) else time_test[plant_mask],
        y=stress_probs[plant_mask],
        mode='lines',
        name=f"Subject: {plant}",
        opacity=0.85,
        line=dict(width=1.5)
    ))

# Add 0.5 Decision Boundary
fig.add_hline(
    y=0.5, line_dash="dash", line_color="black", opacity=0.5,
    annotation_text="Algorithm Decision Boundary (0.5)",
    annotation_position="bottom right"
)

# Add True Stress Onset Line (Decoupled Annotation)
fig.add_vline(
    x=STRESS_ONSET_DATE, line_dash="dash", line_color="red", opacity=0.7
)

fig.add_annotation(
    x=STRESS_ONSET_DATE,
    y=1.0,
    yref="paper",
    text="True Unilateral Biological Stress Onset",
    showarrow=False,
    xanchor="right",
    yanchor="bottom"
)

fig.update_layout(
    title="Unilateral Stress Probability Timeline (Model Confidence)",
    xaxis_title="Timestamp",
    yaxis_title="Probability of Stress [0.0 - 1.0]",
    yaxis_range=[-0.05, 1.05],  # Slight padding for visual clarity
    template="plotly_white",
    hovermode="x unified"
)

output_html = BASE_DIRECTORY + "/stress_probability_output.html"
fig.write_html(output_html, auto_open=True)
print(f"STATUS: Interactive probability plot saved to {output_html}")

# Serialization
joblib.dump(extractor, 'unilateral_extractor.joblib')
joblib.dump(expert_layer, 'unilateral_experts.joblib')
joblib.dump(meta_model, 'unilateral_meta_model.joblib')
print("Unilateral pipeline execution sequence terminated.")