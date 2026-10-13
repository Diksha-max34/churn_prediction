"""Data loading and cleaning for the Telco churn project."""
import pandas as pd

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_data(path: str = "data/telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    # TotalCharges has blank strings for brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    # Collapse "No internet service" / "No phone service" into "No"
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    return df


def feature_columns(df: pd.DataFrame):
    X_cols = [c for c in df.columns if c != TARGET]
    cat = [c for c in X_cols if c not in NUMERIC]
    return NUMERIC, cat
