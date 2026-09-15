from pathlib import Path
import pandas as pd

def read_table(path):
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError("Input must be CSV, XLSX, or XLS.")

def columns(text):
    result = [x.strip() for x in text.split(",") if x.strip()]
    if not result:
        raise ValueError("At least one feature column is required.")
    return result
