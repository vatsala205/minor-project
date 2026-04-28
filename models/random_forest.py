import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from utils.preprocessing import load_data

def train_random_forest():
    df = load_data()
    df = df.dropna(subset=["Risk_Level"])

    X = df.select_dtypes(include=['float64', 'int64']).copy()
    y = df["Risk_Level"]

    X = X.fillna(X.median())

    from imblearn.over_sampling import SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    return {
        "accuracy": accuracy,
        "report": report,
        "feature_importances": feature_importances
    }


if __name__ == "__main__":
    train_random_forest()