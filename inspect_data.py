from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parent


def inspect_parquet(filename: str) -> None:
    path = PROJECT_DIR / filename
    parquet = pq.ParquetFile(path)

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print(f"Size: {path.stat().st_size / 1_000_000:.1f} MB")
    print(f"Rows: {parquet.metadata.num_rows:,}")
    print(f"Columns: {parquet.metadata.num_columns}")
    print("\nSchema:")
    print(parquet.schema_arrow)

    # Read only the first row group for an initial preview.
    sample = parquet.read_row_group(0).to_pandas()

    print("\nFirst five rows:")
    print(sample.head().to_string())

    print("\nMissing values in first row group:")
    print(sample.isna().sum().to_string())

    print("\nUnique values per column in first row group:")
    print(sample.nunique().to_string())


def inspect_labels(filename: str) -> None:
    path = PROJECT_DIR / filename
    labels = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print(f"Rows: {len(labels):,}")
    print("\nColumns and types:")
    print(labels.dtypes.to_string())

    print("\nFirst five rows:")
    print(labels.head().to_string(index=False))

    print("\nMissing values:")
    print(labels.isna().sum().to_string())

    print("\nLabel distribution:")
    print(labels["LABEL"].value_counts(dropna=False).to_string())

    print(f"\nUnique ride IDs: {labels['RIDE_ID'].nunique():,}")
    print(f"Duplicate ride IDs: {labels['RIDE_ID'].duplicated().sum():,}")


if __name__ == "__main__":
    inspect_labels("GBG500_labels.csv")
    inspect_parquet("GBG500.parquet")
    inspect_parquet("SRV3490.parquet")