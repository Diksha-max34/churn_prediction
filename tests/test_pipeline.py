import pandas as pd
from src.data import clean, load_data


def test_clean_handles_blank_total_charges():
    df = pd.DataFrame({"customerID": ["1"], "TotalCharges": [" "], "Churn": ["Yes"],
                       "OnlineSecurity": ["No internet service"]})
    out = clean(df)
    assert out.loc[0, "TotalCharges"] == 0.0
    assert out.loc[0, "Churn"] == 1
    assert out.loc[0, "OnlineSecurity"] == "No"
    assert "customerID" not in out.columns


def test_dataset_shape_and_target():
    df = load_data()
    assert df.shape[0] > 7000
    assert set(df["Churn"].unique()) == {0, 1}
    assert df.isna().sum().sum() == 0
