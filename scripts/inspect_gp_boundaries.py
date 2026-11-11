from pathlib import Path
import pandas as pd

path = Path("data/raw/LGD_panchayats.parquet")

df = pd.read_parquet(path)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(3).to_string())

# Search available fields for the target Panchayat.
for col in df.columns:
    if any(word in col.lower() for word in ("name", "code", "district", "block")):
        print(f"\n--- {col} ---")
        print(
            df.loc[
                df[col].astype(str).str.contains(
                    "Mohmadpur Deopar|Samastipur|Pusa",
                    case=False,
                    regex=True,
                    na=False,
                ),
                [col],
            ].drop_duplicates().head(30).to_string(index=False)
        )