from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

FEATURE_FILE = Path("data/features/features.csv")
MODEL_DIR = Path("models/xgboost")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(["timestamp", "point_id"]).reset_index(drop=True)


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
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

TARGET = "risk_class"


X = df[FEATURES]
y = df[TARGET]


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df[df["timestamp"].dt.year <= 2021]
val_df = df[df["timestamp"].dt.year == 2022]
test_df = df[df["timestamp"].dt.year == 2023]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


print("\nDataset split")
print("-----------------------------")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")


print("\nTraining classes")
print(y_train.value_counts())

print("\nValidation classes")
print(y_val.value_counts())

print("\nTest classes")
print(y_test.value_counts())


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print("\nLabel mapping")

for label, encoded in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(f"{label} -> {encoded}")


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)


print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train_encoded,
    eval_set=[(X_val, y_val_encoded)],
    verbose=False
)


# ============================================================
# VALIDATION
# ============================================================

val_predictions = model.predict(X_val)

print("\n=============================")
print("VALIDATION RESULTS")
print("=============================")

print(
    "Accuracy:",
    round(accuracy_score(y_val_encoded, val_predictions), 4)
)

print(
    classification_report(
        y_val_encoded,
        val_predictions,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# TEST
# ============================================================

test_predictions = model.predict(X_test)

print("\n=============================")
print("TEST RESULTS - 2023")
print("=============================")

print(
    "Accuracy:",
    round(accuracy_score(y_test_encoded, test_predictions), 4)
)

print(
    classification_report(
        y_test_encoded,
        test_predictions,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance")
print(importance_df.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "xgboost.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

joblib.dump(model, model_path)
joblib.dump(label_encoder, encoder_path)

print("\nModel saved to:")
print(model_path)

print("Label encoder saved to:")
print(encoder_path)