import pandas as pd
import os

def normalize_numeric(series):
    return 1 + 4 * (series - series.min()) / (series.max() - series.min())

def standardize_text(series):
    return series.astype(str).str.strip().str.lower()

def map_severity(value, mapping):
    return mapping.get(value, None)

def process_column(df, column, mapping):
    df[column] = standardize_text(df[column])

    # Try numeric conversion
    numeric_version = pd.to_numeric(df[column], errors='coerce')

    if numeric_version.notna().sum() > 0:
        # If numeric values exist, normalize them
        df[column] = numeric_version
        df[column] = normalize_numeric(df[column])
    else:
        # Otherwise map text severity
        df[column] = df[column].apply(lambda x: map_severity(x, mapping))

    return df

def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "data", "Behavioral_Health_Dataset.xlsx")

    df = pd.read_excel(file_path)
    df = df.drop(columns=["Unnamed: 22", "Unnamed: 23"])

    mappings = {
        "Sleep_Quality": {
            "very poor": 5, "poor": 4, "average": 3,
            "good": 2, "very good": 1
        },
        "Stress_Level": {
            "not at all": 1, "slightly": 2,
            "moderately": 3, "very": 4, "extremely": 5
        },
        "Mentally_Drained": {
            "never": 1, "occasionally": 2,
            "often": 4, "almost always": 5
        },
        "Social_Connectedness": {
            "very connected": 1, "somewhat connected": 2,
            "neutral": 3, "somewhat disconnected": 4,
            "very disconnected": 5
        },
        "Wake_Up_Feeling": {
            "refreshed": 1, "slightly tired": 2,
            "tired": 4, "exhausted": 5,
            "groggy": 3, "restless": 3
        },
        "Daytime_Sleepiness": {
            "never": 1, "rarely": 2,
            "sometimes": 3, "often": 4,
            "almost always": 5
        },
        "Sleep_Consistency": {
            "very consistent": 1, "consistent": 2,
            "mostly consistent": 2,
            "somewhat inconsistent": 3,
            "slightly irregular": 3,
            "very inconsistent": 5,
            "very irregular": 5
        },
        "Workload_Demand": {
            "very low": 1, "low": 2,
            "moderate": 3, "high": 4,
            "very high": 5
        },
        "Resting_HR": {
            "below 60 bpm": 1,
            "60-70 bpm": 2,
            "70-80 bpm": 3,
            "above 80 bpm": 5,
            "not sure": 3
        }
    }

    risk_columns = list(mappings.keys())

    for col in risk_columns:
        df = process_column(df, col, mappings[col])

    # Fill any remaining missing with median severity
    df[risk_columns] = df[risk_columns].fillna(df[risk_columns].median())

    # Weighted risk score
    risk_score = (
        2 * df["Stress_Level"] +
        2 * df["Mentally_Drained"] +
        2 * df["Workload_Demand"] +
        1.5 * df["Sleep_Quality"] +
        1.5 * df["Sleep_Consistency"] +
        1.5 * df["Daytime_Sleepiness"] +
        1.5 * df["Wake_Up_Feeling"] +
        1.5 * df["Social_Connectedness"] +
        1 * df["Resting_HR"]
    )

    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

    df["Risk_Level"] = pd.cut(
        risk_score,
        bins=[0, 0.33, 0.66, 1],
        labels=["Low Risk", "Mid Risk", "High Risk"]
    )


    return df

if __name__ == "__main__":
    load_data()