from pathlib import Path

import pandas as pd
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data/processed/insar_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "features.csv"


# Offline training path: data/processed/insar_cleaned.csv -> prepare_features()
# -> data/features/features.csv. Live inference calls prepare_features() on
# the raw InSAR data directly (see src/inference/predict.py).
# --------------------------------------------------
# Features used by the ML models
# --------------------------------------------------

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


# --------------------------------------------------
# Main feature engineering
# --------------------------------------------------

def prepare_features(df):
    df = df.copy()
    print("Original shape:", df.shape)


    # --------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )


    # Remove rows where timestamp is missing

    df = df.dropna(
        subset=["timestamp"]
    )


    # --------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------

    df = df.sort_values(
        ["point_id", "timestamp"]
    ).reset_index(drop=True)


    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = FEATURES + [
        TARGET,
        "timestamp",
        "point_id"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


    # --------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------

    output_columns = [
        "point_id",
        "timestamp"
    ] + FEATURES + [
        TARGET
    ]

    features_df = df[output_columns].copy()


    # --------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------

    features_df = features_df.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------
    # Convert feature columns to numeric
    # --------------------------------------------------

    for column in FEATURES:

        features_df[column] = pd.to_numeric(
            features_df[column],
            errors="coerce"
        )


    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\nMissing values before filling:")

    print(
        features_df[FEATURES]
        .isna()
        .sum()
        .to_string()
    )


    for column in FEATURES:
        median_value = features_df[column].median()
        if pd.isna(median_value):
            raise ValueError(
                f"Cannot preprocess {column}: no valid values are available."
            )
        features_df[column] = features_df[column].fillna(
            median_value
        )


    # --------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------

    features_df = features_df.dropna(
        subset=[TARGET]
    )


    # --------------------------------------------------
    # Final sorting
    # --------------------------------------------------

    features_df = features_df.sort_values(
        ["timestamp", "point_id"]
    ).reset_index(drop=True)


    return features_df


def create_features():
    print("\nLoading cleaned InSAR data...")
    features_df = prepare_features(pd.read_csv(INPUT_FILE))

    features_df.to_csv(OUTPUT_FILE, index=False)


    print("\n================================")
    print("Feature engineering completed")
    print("================================")

    print("Final shape:", features_df.shape)

    print("\nFeatures used:")

    for feature in FEATURES:
        print(" -", feature)

    print("\nTarget:")
    print(" -", TARGET)

    print("\nRisk distribution:")

    print(
        features_df[TARGET]
        .value_counts()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    create_features()