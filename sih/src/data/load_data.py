# src/data/load_data.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Runtime path: predict_latest() in src/inference/predict.py loads the raw
# InSAR CSV from data/raw/insar/ through this function.
def load_insar(verbose=True):
    files = sorted((PROJECT_ROOT / "data/raw/insar").glob("*.csv"))

    if not files:
        raise FileNotFoundError("No InSAR CSV found.")

    df = pd.read_csv(files[0])

    if verbose:
        print("InSAR shape:", df.shape)
        print("InSAR columns:")
        print(df.columns.tolist())

    return df


# Seismic data is currently included in the API response as metadata; the
# trained prediction models use the InSAR features, not this dataset.
def load_seismic(verbose=True):
    data_dir = PROJECT_ROOT / "data/raw/seismic"
    files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.xls*"))

    if not files:
        raise FileNotFoundError("No seismic CSV or Excel file found.")

    file = files[0]
    if file.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    if verbose:
        print("Seismic shape:", df.shape)
        print("Seismic columns:")
        print(df.columns.tolist())

    return df


if __name__ == "__main__":
    insar = load_insar()
    seismic = load_seismic()