import pandas as pd
import sqlite3
from pathlib import Path

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "telco_churn.csv"
DB_PATH = DATA_DIR / "telco.db"
TABLE = "customers"

def to_snake(s: str) -> str:
    return (
        s.strip()
         .replace(" ", "_")
         .replace("-", "_")
         .replace("/", "_")
         .replace("__", "_")
         .lower()
    )

def main():
    assert CSV_PATH.exists(), f"CSV not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)

    # Standardize column names (snake_case)
    df.columns = [to_snake(c) for c in df.columns]

    # Coerce numeric fields
    for col in ["tenure", "monthly_charges", "total_charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize churn values (Yes/No)
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(str).str.strip().str.title()

    # Build SQLite DB fresh each time
    DATA_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_churn ON {TABLE}(churn)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_contract ON {TABLE}(contract)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_paymentmethod ON {TABLE}(paymentmethod)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_tenure ON {TABLE}(tenure)")

        # Quick sanity checks
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        print(f"Loaded {n} rows into table '{TABLE}' at {DB_PATH}")
        cols = [d[0] for d in conn.execute(f"SELECT * FROM {TABLE} LIMIT 1").description]
        print("Columns:", cols)

if __name__ == "__main__":
    main()
