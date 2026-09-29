
from functools import lru_cache
from pathlib import Path
from collections import deque

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.load_data import load_insar, load_seismic
from src.features.feature_engineering import FEATURES, prepare_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRS = {
    "xgboost": PROJECT_ROOT / "models/xgboost",
    "random_forest": PROJECT_ROOT / "models/random_forest",
    "attention": PROJECT_ROOT / "models/attention",
}

TRAINING_FEATURE_FILE = PROJECT_ROOT / "data/features/features.csv"

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

SENSOR_FEATURES = [
    "tilt",
    "displacement",
    "crack_width",
    "strain",
]

# Keep recent sensor readings
_sensor_history = deque(maxlen=20)


RISK_NORMALIZATION = {
    "LOW": "LOW",
    "MODERATE": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "HIGH",
}


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
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, values):

        values = self.input_projection(values)

        values = self.transformer(values)

        return self.classifier(
            values[:, -1, :]
        )


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@lru_cache(maxsize=1)
def _load_models():

    xgb_dir = MODEL_DIRS["xgboost"]

    models = {
        "xgboost": (
            joblib.load(
                xgb_dir / "xgboost.pkl"
            ),
            joblib.load(
                xgb_dir / "label_encoder.pkl"
            ),
        )
    }

    errors = {}

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    try:

        rf_dir = MODEL_DIRS["random_forest"]

        models["random_forest"] = (
            joblib.load(
                rf_dir / "random_forest.pkl"
            ),
            joblib.load(
                rf_dir / "label_encoder.pkl"
            ),
        )

    except Exception as error:

        errors["random_forest"] = str(error)

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    try:

        attention_dir = MODEL_DIRS["attention"]

        scaler = joblib.load(
            attention_dir / "scaler.pkl"
        )

        encoder = joblib.load(
            attention_dir / "label_encoder.pkl"
        )

        model = AttentionModel(
            len(FEATURES),
            len(encoder.classes_)
        )

        state = torch.load(
            attention_dir / "attention_model.pt",
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state)

        model.eval()

        if list(
            getattr(
                scaler,
                "feature_names_in_",
                FEATURES
            )
        ) != list(FEATURES):

            raise ValueError(
                "Attention scaler feature schema "
                "does not match feature engineering."
            )

        models["attention"] = (
            model,
            encoder,
            scaler
        )

    except Exception as error:

        errors["attention"] = str(error)

    # --------------------------------------------------------
    # Verify feature schema
    # --------------------------------------------------------

    for name in (
        "xgboost",
        "random_forest"
    ):

        if name not in models:
            continue

        model = models[name][0]

        model_features = getattr(
            model,
            "feature_names_in_",
            FEATURES
        )

        if list(model_features) != list(FEATURES):

            raise ValueError(
                f"{name} feature schema "
                "does not match feature engineering."
            )

    return models, errors


# ============================================================
# RISK → ACTION
# ============================================================

RISK_ACTIONS = {

    "LOW": {
        "action": "NORMAL",
        "led": "GREEN",
        "message": "Ground conditions are stable."
    },

    "MEDIUM": {
        "action": "MONITOR",
        "led": "YELLOW",
        "message": "Increased monitoring recommended."
    },

    "HIGH": {
        "action": "WARNING",
        "led": "RED",
        "message": "Potential subsidence risk detected."
    },

}


# ============================================================
# RISK HELPERS
# ============================================================

def _normalise_risk(label):

    try:

        return RISK_NORMALIZATION[
            str(label).upper()
        ]

    except KeyError as error:

        raise ValueError(
            f"Unsupported model risk class: {label}"
        ) from error


def _canonical_probabilities(
    labels,
    probabilities
):

    result = {
        risk: 0.0
        for risk in RISK_LEVELS
    }

    for label, probability in zip(
        labels,
        probabilities
    ):

        result[
            _normalise_risk(label)
        ] += float(probability)

    return result


def _tree_prediction(
    model,
    encoder,
    values
):

    encoded_prediction = int(
        model.predict(values)[0]
    )

    native_prediction = str(
        encoder.inverse_transform(
            [encoded_prediction]
        )[0]
    )

    encoded_classes = getattr(
        model,
        "classes_",
        range(len(encoder.classes_))
    )

    native_classes = [

        str(
            encoder.inverse_transform(
                [int(value)]
            )[0]
        )

        for value in encoded_classes
    ]

    probabilities = model.predict_proba(
        values
    )[0]

    return (
        native_prediction,
        _canonical_probabilities(
            native_classes,
            probabilities
        )
    )


def _model_result(
    native_prediction,
    probabilities
):

    return {

        "prediction":
            _normalise_risk(
                native_prediction
            ),

        "native_prediction":
            native_prediction,

        "confidence":
            round(
                max(probabilities.values()),
                6
            ),

        "probabilities": {
            key: round(value, 6)
            for key, value
            in probabilities.items()
        },
    }


# ============================================================
# GET REAL TRAINING MEDIANS
# ============================================================

@lru_cache(maxsize=1)
def _get_default_features():

    if not TRAINING_FEATURE_FILE.exists():

        raise FileNotFoundError(
            "Training feature file not found: "
            + str(TRAINING_FEATURE_FILE)
        )

    df = pd.read_csv(
        TRAINING_FEATURE_FILE
    )

    defaults = {}

    for feature in FEATURES:

        if feature not in df.columns:

            raise ValueError(
                f"Training feature missing: {feature}"
            )

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        median_value = values.median()

        if pd.isna(median_value):

            raise ValueError(
                f"No valid training values for {feature}"
            )

        defaults[feature] = float(
            median_value
        )

    return defaults


# ============================================================
# 4 SENSOR VALUES → 18 MODEL FEATURES
# ============================================================

def sensor_to_model_features(
    tilt,
    displacement,
    crack_width,
    strain
):

    tilt = float(tilt)
    displacement = float(displacement)
    crack_width = float(crack_width)
    strain = float(strain)

    # --------------------------------------------------------
    # Store current sensor reading
    # --------------------------------------------------------

    current = {
        "tilt": tilt,
        "displacement": displacement,
        "crack_width": crack_width,
        "strain": strain,
    }

    _sensor_history.append(current)

    # --------------------------------------------------------
    # Start with REAL training-data medians
    # --------------------------------------------------------

    values = _get_default_features().copy()

    # --------------------------------------------------------
    # Displacement-derived features
    # --------------------------------------------------------

    displacement_values = [
        item["displacement"]
        for item in _sensor_history
    ]

    current_displacement = displacement_values[-1]

    # Current displacement
    values[
        "cumulative_displacement_mm"
    ] = current_displacement

    # LOS displacement is NOT automatically changed
    # because generic ESP32 displacement is not necessarily
    # InSAR line-of-sight displacement.

    values[
        "los_displacement_mm"
    ] = _get_default_features()[
        "los_displacement_mm"
    ]

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    if len(displacement_values) >= 2:

        velocity = (
            displacement_values[-1]
            - displacement_values[-2]
        )

    else:

        velocity = 0.0

    values[
        "displacement_velocity_mm_month"
    ] = velocity

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    if len(displacement_values) >= 3:

        previous_velocity = (
            displacement_values[-2]
            - displacement_values[-3]
        )

        acceleration = (
            velocity
            - previous_velocity
        )

    else:

        acceleration = 0.0

    values[
        "displacement_acceleration"
    ] = acceleration

    # --------------------------------------------------------
    # Rolling displacement
    # --------------------------------------------------------

    last_displacements = displacement_values[-3:]

    values[
        "rolling_mean_displacement_3m"
    ] = float(
        np.mean(last_displacements)
    )

    # --------------------------------------------------------
    # Rolling velocity
    # --------------------------------------------------------

    if len(displacement_values) >= 2:

        velocities = np.diff(
            displacement_values
        )

        last_velocities = velocities[-3:]

        values[
            "rolling_velocity_3m"
        ] = float(
            np.mean(last_velocities)
        )

    else:

        values[
            "rolling_velocity_3m"
        ] = 0.0

    # --------------------------------------------------------
    # Deformation trend
    # --------------------------------------------------------

    if len(displacement_values) >= 2:

        if (
            displacement_values[-1]
            > displacement_values[0]
        ):

            trend = 1.0

        elif (
            displacement_values[-1]
            < displacement_values[0]
        ):

            trend = -1.0

        else:

            trend = 0.0

    else:

        trend = 0.0

    values[
        "deformation_trend"
    ] = trend

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # tilt, crack_width and strain are NOT mapped to
    # unrelated InSAR/environment features.
    #
    # They are accepted by the API, but the current
    # XGBoost model was NOT trained on these variables.
    #
    # Therefore they cannot directly influence the
    # learned prediction until the model is retrained
    # using real labelled sensor data.
    #
    # --------------------------------------------------------

    result = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    return result


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_risk(input_data):

    """
    Predict mine subsidence risk.

    Accepted input:

    1. Four sensor values:

       {
           "tilt": ...,
           "displacement": ...,
           "crack_width": ...,
           "strain": ...
       }

    OR

    2. Full existing 18-feature model input.
    """

    # --------------------------------------------------------
    # Dictionary → DataFrame
    # --------------------------------------------------------

    if isinstance(
        input_data,
        dict
    ):

        # ----------------------------------------------------
        # NEW SENSOR INPUT
        # ----------------------------------------------------

        if all(
            feature in input_data
            for feature in SENSOR_FEATURES
        ):

            input_data = sensor_to_model_features(
                tilt=input_data["tilt"],
                displacement=input_data["displacement"],
                crack_width=input_data["crack_width"],
                strain=input_data["strain"],
            )

        # ----------------------------------------------------
        # EXISTING 18-FEATURE INPUT
        # ----------------------------------------------------

        else:

            input_data = pd.DataFrame(
                [input_data]
            )

    elif not isinstance(
        input_data,
        pd.DataFrame
    ):

        raise TypeError(
            "input_data must be a dictionary "
            "or pandas DataFrame"
        )

    # --------------------------------------------------------
    # Check model features
    # --------------------------------------------------------

    missing_features = [

        feature

        for feature in FEATURES

        if feature not in input_data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            + ", ".join(
                missing_features
            )
        )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    values = input_data.loc[
        :,
        FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if (
        values.isna().any().any()
        or
        not np.isfinite(
            values.to_numpy(
                dtype=float
            )
        ).all()
    ):

        raise ValueError(
            "Prediction input contains "
            "missing or non-finite model features."
        )

    # --------------------------------------------------------
    # Load XGBoost
    # --------------------------------------------------------

    models, _ = _load_models()

    model, label_encoder = models[
        "xgboost"
    ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    native_prediction, probabilities = _tree_prediction(
        model,
        label_encoder,
        values
    )

    risk = max(
        probabilities,
        key=probabilities.get
    )

    action = RISK_ACTIONS[
        risk
    ]

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "risk":
            risk,

        "risk_level":
            risk,

        "native_risk":
            native_prediction,

        "confidence":
            round(
                probabilities[risk],
                6
            ),

        "probabilities": {
            key: round(
                value,
                6
            )

            for key, value
            in probabilities.items()
        },

        "action":
            action["action"],

        "action_signal":
            action["led"],

        "led_signal":
            action["led"],

        "message":
            action["message"],
    }


# ============================================================
# EXISTING /predict/latest
# ============================================================

def predict_latest():

    raw_insar = load_insar(
        verbose=False
    )

    raw_seismic = load_seismic(
        verbose=False
    )

    features = prepare_features(
        raw_insar
    )

    if features.empty:

        raise ValueError(
            "No valid InSAR observations "
            "remain after preprocessing."
        )

    features = features.sort_values(
        [
            "timestamp",
            "point_id"
        ]
    ).reset_index(
        drop=True
    )

    latest = features.iloc[-1]

    latest_year = latest[
        "timestamp"
    ].year

    sequence = features[
        (
            features["point_id"]
            == latest["point_id"]
        )
        &
        (
            features["timestamp"].dt.year
            == latest_year
        )
    ].tail(8)

    if len(sequence) < 8:

        raise ValueError(
            "Attention inference requires "
            "8 observations in the latest year; "
            f"found {len(sequence)}."
        )

    latest_values = features.iloc[
        [-1]
    ][FEATURES].astype(float)

    models, model_errors = _load_models()

    per_model = {}

    probability_sets = []

    # --------------------------------------------------------
    # XGBoost + Random Forest
    # --------------------------------------------------------

    for name in (
        "xgboost",
        "random_forest"
    ):

        if name not in models:
            continue

        model, encoder = models[
            name
        ]

        native_prediction, probabilities = _tree_prediction(
            model,
            encoder,
            latest_values
        )

        per_model[name] = _model_result(
            native_prediction,
            probabilities
        )

        probability_sets.append(
            probabilities
        )

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    if "attention" in models:

        model, encoder, scaler = models[
            "attention"
        ]

        scaled_sequence = scaler.transform(
            sequence[FEATURES]
        )

        values = torch.tensor(
            scaled_sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.inference_mode():

            attention_probabilities = torch.softmax(
                model(values),
                dim=1
            )[0].cpu().numpy()

        native_classes = [
            str(label)
            for label in encoder.classes_
        ]

        native_prediction = native_classes[
            int(
                np.argmax(
                    attention_probabilities
                )
            )
        ]

        probabilities = _canonical_probabilities(
            native_classes,
            attention_probabilities
        )

        per_model["attention"] = _model_result(
            native_prediction,
            probabilities
        )

        probability_sets.append(
            probabilities
        )

    if not probability_sets:

        raise RuntimeError(
            "No compatible trained models "
            "are available for inference."
        )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ensemble = {

        risk: float(
            np.mean(
                [
                    probabilities[risk]
                    for probabilities
                    in probability_sets
                ]
            )
        )

        for risk in RISK_LEVELS
    }

    risk = max(
        ensemble,
        key=ensemble.get
    )

    # --------------------------------------------------------
    # Seismic metadata
    # --------------------------------------------------------

    seismic_class_counts = (

        {
            str(label): int(count)

            for label, count
            in raw_seismic["class"]
            .value_counts(
                dropna=False
            ).items()
        }

        if "class"
        in raw_seismic.columns

        else {}
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "system":
            "Mine Subsidence AI",

        "risk":
            risk,

        "confidence":
            round(
                ensemble[risk],
                6
            ),

        "probabilities": {
            key: round(
                value,
                6
            )

            for key, value
            in ensemble.items()
        },

        "models":
            per_model,

        "model_errors":
            model_errors,

        "data_source":
            "Real InSAR; real seismic dataset loaded separately",

        "data": {

            "point_id":
                str(
                    latest["point_id"]
                ),

            "timestamp":
                latest[
                    "timestamp"
                ].isoformat(),

            "insar_observations":
                int(
                    len(raw_insar)
                ),

            "seismic_records":
                int(
                    len(raw_seismic)
                ),

            "seismic_class_counts":
                seismic_class_counts,

            "seismic_used_for_model":
                False,

            "seismic_note":
                "The seismic dataset has no timestamp, "
                "location, or point ID to align with InSAR, "
                "and the saved models were trained on "
                "InSAR features only.",

            "attention_sequence_length":
                int(
                    len(sequence)
                ),
        },
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sensor_data = {

        "tilt": 2.5,

        "displacement": 12.0,

        "crack_width": 1.2,

        "strain": 0.003,
    }

    print(
        predict_risk(
            test_sensor_data
        )
    )

