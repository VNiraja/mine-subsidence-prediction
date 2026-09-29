import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "data/processed/insar_cleaned.csv"
MODEL_PATH = "models/random_forest/random_forest.pkl"
LABEL_PATH = "models/random_forest/label_encoder.pkl"


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


# -----------------------------
# 1. Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# -----------------------------
# 2. Select features
# -----------------------------

X = df[FEATURES].copy()
y = df[TARGET].copy()


# -----------------------------
# 3. Handle missing values
# -----------------------------

X = X.fillna(X.median())


# -----------------------------
# 4. Chronological split
# -----------------------------

train = df[df["timestamp"].dt.year <= 2021]
validation = df[df["timestamp"].dt.year == 2022]
test = df[df["timestamp"].dt.year == 2023]


X_train = train[FEATURES].fillna(X.median())
y_train = train[TARGET]

X_val = validation[FEATURES].fillna(X.median())
y_val = validation[TARGET]

X_test = test[FEATURES].fillna(X.median())
y_test = test[TARGET]


print("\nDataset split")
print("-----------------------------")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Test       :", len(X_test))


print("\nTraining classes")
print(y_train.value_counts())


print("\nValidation classes")
print(y_val.value_counts())


print("\nTest classes")
print(y_test.value_counts())


# -----------------------------
# 5. Encode labels
# -----------------------------

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)
y_test_encoded = encoder.transform(y_test)


print("\nLabel mapping")
for label, number in zip(
    encoder.classes_,
    range(len(encoder.classes_))
):
    print(label, "->", number)


# -----------------------------
# 6. Train Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_encoded)


# -----------------------------
# 7. Validation evaluation
# -----------------------------

val_predictions = model.predict(X_val)

print("\n=============================")
print("VALIDATION RESULTS")
print("=============================")

print(
    "Accuracy:",
    accuracy_score(y_val_encoded, val_predictions)
)

print(
    classification_report(
        y_val_encoded,
        val_predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# -----------------------------
# 8. Final test evaluation
# -----------------------------

test_predictions = model.predict(X_test)

print("\n=============================")
print("TEST RESULTS - 2023")
print("=============================")

print(
    "Accuracy:",
    accuracy_score(y_test_encoded, test_predictions)
)

print(
    classification_report(
        y_test_encoded,
        test_predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# -----------------------------
# 9. Feature importance
# -----------------------------

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance")
print(importance.to_string(index=False))


# -----------------------------
# 10. Save model
# -----------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, LABEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("Label encoder saved to:")
print(LABEL_PATH)