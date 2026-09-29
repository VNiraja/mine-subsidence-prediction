# src/data/load_data.py

import pandas as pd
from pathlib import Path


# When run as a script, this offline step reads data/raw/insar and
# data/raw/seismic, removes duplicate rows, and writes data/processed/*.csv.
# Live inference reads the raw InSAR input separately via src/data/load_data.py.
def load_insar():
    files = list(Path("data/raw/insar").glob("*.csv"))

    if not files:
        raise FileNotFoundError("No InSAR CSV found.")

    df = pd.read_csv(files[0])

    print("InSAR shape:", df.shape)
    print("InSAR columns:")
    print(df.columns.tolist())

    return df


def load_seismic():
    data_dir = Path("data/raw/seismic")
    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xls*"))

    if not files:
        raise FileNotFoundError("No seismic CSV or Excel file found.")

    file = files[0]
    if file.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    print("Seismic shape:", df.shape)
    print("Seismic columns:")
    print(df.columns.tolist())

    return df


if __name__ == "__main__":
    insar = load_insar().drop_duplicates()
    seismic = load_seismic().drop_duplicates()

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    insar.to_csv(processed_dir / "insar_cleaned.csv", index=False)
    seismic.to_csv(processed_dir / "seismic_cleaned.csv", index=False)

    print("Saved cleaned datasets to:", processed_dir)