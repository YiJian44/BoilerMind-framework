import pandas as pd
from pathlib import Path

path = Path(
    r"D:\YANGZIYU\2026AIscientist\ai\boiler_soft_sensor_models\data\boiler_181var.xlsx"
)

df = pd.read_excel(path, nrows=3, engine="openpyxl")

print("FILE =", path)
print("RAW_COLUMN_COUNT =", len(df.columns))
print()

for i, col in enumerate(df.columns, start=1):
    print(f"{i:03d} | {col}")
