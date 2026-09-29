from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

FEATURE_FILE = Path("data/features/features.csv")
MODEL_DIR = Path("models/attention")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 8
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# FEATURES
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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["point_id", "timestamp"]
).reset_index(drop=True)


print("\nDataset shape:", df.shape)
print("Number of points:", df["point_id"].nunique())


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["target_encoded"] = label_encoder.fit_transform(
    df[TARGET]
)

print("\nLabel mapping")

for label, encoded in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(f"{label} -> {encoded}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df[df["timestamp"].dt.year <= 2021].copy()
val_df = df[df["timestamp"].dt.year == 2022].copy()
test_df = df[df["timestamp"].dt.year == 2023].copy()


print("\nDataset split")
print("-----------------------------")
print(f"Training rows   : {len(train_df)}")
print(f"Validation rows : {len(val_df)}")
print(f"Test rows       : {len(test_df)}")


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

scaler.fit(train_df[FEATURES])

train_df.loc[:, FEATURES] = scaler.transform(
    train_df[FEATURES]
)

val_df.loc[:, FEATURES] = scaler.transform(
    val_df[FEATURES]
)

test_df.loc[:, FEATURES] = scaler.transform(
    test_df[FEATURES]
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(dataframe):

    X_sequences = []
    y_targets = []

    for point_id, group in dataframe.groupby("point_id"):

        group = group.sort_values("timestamp")

        values = group[FEATURES].values
        targets = group["target_encoded"].values

        for i in range(
            SEQUENCE_LENGTH - 1,
            len(group)
        ):

            start = i - SEQUENCE_LENGTH + 1

            sequence = values[start:i + 1]

            target = targets[i]

            X_sequences.append(sequence)
            y_targets.append(target)

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_targets, dtype=np.int64)
    )


X_train, y_train = create_sequences(train_df)
X_val, y_val = create_sequences(val_df)
X_test, y_test = create_sequences(test_df)


print("\nSequence shapes")
print("-----------------------------")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)


# ============================================================
# ATTENTION MODEL
# ============================================================

class AttentionModel(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes,
        d_model=64,
        num_heads=4,
        num_layers=2
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        # Use the final timestep representation
        x = x[:, -1, :]

        output = self.classifier(x)

        return output


# ============================================================
# MODEL
# ============================================================

input_size = len(FEATURES)

num_classes = len(label_encoder.classes_)

model = AttentionModel(
    input_size=input_size,
    num_classes=num_classes
)


print("\nAttention model")
print("-----------------------------")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining attention model...")

for epoch in range(EPOCHS):

    model.train()

    permutation = torch.randperm(
        X_train.size(0)
    )

    total_loss = 0

    for i in range(
        0,
        X_train.size(0),
        BATCH_SIZE
    ):

        indices = permutation[
            i:i + BATCH_SIZE
        ]

        batch_X = X_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val)

        val_predictions = torch.argmax(
            val_outputs,
            dim=1
        )

        val_accuracy = accuracy_score(
            y_val.numpy(),
            val_predictions.numpy()
        )

    if (epoch + 1) % 10 == 0:

        avg_loss = (
            total_loss /
            max(1, X_train.size(0) // BATCH_SIZE)
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )


# ============================================================
# VALIDATION RESULTS
# ============================================================

model.eval()

with torch.no_grad():

    val_outputs = model(X_val)

    val_predictions = torch.argmax(
        val_outputs,
        dim=1
    )


print("\n=============================")
print("VALIDATION RESULTS")
print("=============================")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_val.numpy(),
            val_predictions.numpy()
        ),
        4
    )
)

print(
    classification_report(
        y_val.numpy(),
        val_predictions.numpy(),
        labels=range(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# TEST RESULTS
# ============================================================

with torch.no_grad():

    test_outputs = model(X_test)

    test_predictions = torch.argmax(
        test_outputs,
        dim=1
    )


print("\n=============================")
print("TEST RESULTS - 2023")
print("=============================")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test.numpy(),
            test_predictions.numpy()
        ),
        4
    )
)

print(
    classification_report(
        y_test.numpy(),
        test_predictions.numpy(),
        labels=range(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "attention_model.pt"

scaler_path = MODEL_DIR / "scaler.pkl"

encoder_path = MODEL_DIR / "label_encoder.pkl"


torch.save(
    model.state_dict(),
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    label_encoder,
    encoder_path
)


print("\nModel saved to:")
print(model_path)

print("Scaler saved to:")
print(scaler_path)

print("Label encoder saved to:")
print(encoder_path)