import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv("data/features/features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])


features = [
    "latitude",
    "longitude",
    "cumulative_displacement_mm",
    "displacement_velocity_mm_month",
    "displacement_acceleration",
    "los_displacement_mm",
    "coherence",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "rainfall_mm",
    "cumulative_rainfall_mm",
    "mining_factor",
    "distance_to_centre_deg",
    "rolling_mean_displacement_3m",
    "rolling_velocity_3m",
    "deformation_trend",
    "rainfall_anomaly"
]


print("Dataset shape:", df.shape)


# ============================================================
# 2. LABEL MAPPING
# ============================================================

label_map = {
    "CRITICAL": 0,
    "HIGH": 1,
    "LOW": 2,
    "MODERATE": 3
}


# ============================================================
# 3. CHRONOLOGICAL TEST DATA
# ============================================================

test_mask = df["timestamp"].dt.year == 2023

X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, "risk_class"].map(label_map)


print("\nTest rows:", len(X_test))


# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\nLoading Random Forest...")

rf_model = joblib.load(
    "models/random_forest/random_forest.pkl"
)

rf_pred = rf_model.predict(X_test)


# ============================================================
# 5. XGBOOST
# ============================================================

print("Loading XGBoost...")

xgb_model = joblib.load(
    "models/xgboost/xgboost.pkl"
)

xgb_pred = xgb_model.predict(X_test)


# ============================================================
# 6. ATTENTION MODEL
# ============================================================

print("Loading Attention model...")


class AttentionModel(torch.nn.Module):

    def __init__(
        self,
        input_dim=18,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=4
    ):
        super().__init__()

        self.input_projection = torch.nn.Linear(
            input_dim,
            d_model
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.classifier(x)

        return x


attention_model = AttentionModel()


attention_model.load_state_dict(
    torch.load(
        "models/attention/attention_model.pt",
        map_location="cpu"
    )
)

attention_model.eval()


# ============================================================
# 7. LOAD ATTENTION SCALER
# ============================================================

scaler = joblib.load(
    "models/attention/scaler.pkl"
)


# ============================================================
# 8. CREATE ATTENTION TEST SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 8

test_df = df[test_mask].copy()

test_df = test_df.sort_values(
    ["point_id", "timestamp"]
)


X_test_attention = []
y_test_attention = []


for point_id, group in test_df.groupby("point_id"):

    group = group.sort_values("timestamp")

    X_group = group[features].values

    y_group = group["risk_class"].map(
        label_map
    ).values

    # Scale using scaler fitted on training data
    X_group = scaler.transform(X_group)

    for i in range(
        len(X_group) - SEQUENCE_LENGTH + 1
    ):

        X_test_attention.append(
            X_group[
                i:i + SEQUENCE_LENGTH
            ]
        )

        y_test_attention.append(
            y_group[
                i + SEQUENCE_LENGTH - 1
            ]
        )


X_test_attention = np.array(
    X_test_attention,
    dtype=np.float32
)

y_test_attention = np.array(
    y_test_attention
)


print("\nAttention test sequences:")
print("X_test:", X_test_attention.shape)
print("y_test:", y_test_attention.shape)


# ============================================================
# 9. ATTENTION PREDICTION
# ============================================================

X_tensor = torch.tensor(
    X_test_attention,
    dtype=torch.float32
)


with torch.no_grad():

    outputs = attention_model(
        X_tensor
    )

    attention_pred = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro Precision : {precision:.4f}"
    )

    print(
        f"Macro Recall    : {recall:.4f}"
    )

    print(
        f"Macro F1        : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=[
                "CRITICAL",
                "HIGH",
                "LOW",
                "MODERATE"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    print(cm)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    }


# ============================================================
# 11. EVALUATE ALL THREE MODELS
# ============================================================

results = []


results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred
    )
)


results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_pred
    )
)


results.append(
    evaluate_model(
        "Attention Transformer",
        y_test_attention,
        attention_pred
    )
)


# ============================================================
# 12. FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    results
)


print("\n\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE COMPARISON
# ============================================================

os.makedirs(
    "models/comparison",
    exist_ok=True
)


comparison.to_csv(
    "models/comparison/model_comparison.csv",
    index=False
)


print("\nComparison saved to:")

print(
    "models/comparison/model_comparison.csv"
)