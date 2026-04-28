import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = "/Users/vatsalasmacbook/PycharmProjects/behavioral_health_ai/data/Behavioral_Health_Dataset.xlsx"

df = pd.read_excel(DATA_PATH)

print("Original Shape:", df.shape)

# ----------------------------
# REMOVE CORRUPTED PART
# ----------------------------
df = df.iloc[135:].reset_index(drop=True)

print("After removing first 135 rows:", df.shape)

# ----------------------------
# CLEAN DATA
# ----------------------------
df = df.drop(columns=["Unnamed: 22", "Unnamed: 23"], errors="ignore")

# Convert Mood_Stability to numeric
df["Mood_Stability"] = pd.to_numeric(df["Mood_Stability"], errors="coerce")

# Drop invalid rows
df = df.dropna()

# ----------------------------
# CREATE LABELS (LOW / MEDIUM / HIGH)
# ----------------------------
def categorize_mood(x):
    if x <= 3:
        return "Low"
    elif x <= 7:
        return "Medium"
    else:
        return "High"

df["Mood_Label"] = df["Mood_Stability"].apply(categorize_mood)

print(df["Mood_Label"].value_counts())

# ----------------------------
# FEATURES & TARGET
# ----------------------------
X = df.drop(columns=["Mood_Stability", "Mood_Label"])
y = df["Mood_Label"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Convert categorical features
X = pd.get_dummies(X)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ----------------------------
# 1. CONFUSION MATRIX
# ----------------------------
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# 2. ROC CURVE (MULTI-CLASS)
# ----------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Class {i})")
    plt.legend()
    plt.show()

# ----------------------------
# 3. PRECISION-RECALL CURVE
# ----------------------------
for i in range(3):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Class {i})")
    plt.show()

# ----------------------------
# 4. FEATURE IMPORTANCE
# ----------------------------
importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure()
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Top Feature Importance")
plt.show()

# ----------------------------
# 5. CLASS DISTRIBUTION
# ----------------------------
plt.figure()
sns.countplot(x=df["Mood_Label"])
plt.title("Class Distribution")
plt.show()

# ----------------------------
# 6. CORRELATION HEATMAP
# ----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr())
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# 7. MOOD DISTRIBUTION
# ----------------------------
plt.figure()
plt.hist(df["Mood_Stability"])
plt.title("Mood Stability Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# ----------------------------
# 8. MOOD TIMELINE
# ----------------------------
mood_scores = df["Mood_Stability"].values[:50]

plt.figure()
plt.plot(mood_scores)
plt.title("Mood Variability Over Time")
plt.xlabel("Time")
plt.ylabel("Mood Score")
plt.show()

# ----------------------------
# 9. MVI (SIMPLIFIED)
# ----------------------------
mvi = np.std(mood_scores) + np.abs(np.diff(mood_scores)).mean()

plt.figure()
plt.plot([mvi]*50)
plt.title("Mood Variability Index (MVI)")
plt.show()

# ----------------------------
# 10. THRESHOLD VS RECALL
# ----------------------------
thresholds = np.linspace(0, 1, 50)
recalls = []

y_binary = (y_test == 2).astype(int)

for t in thresholds:
    preds = (y_prob[:, 2] >= t).astype(int)
    recall = sum((preds == 1) & (y_binary == 1)) / max(sum(y_binary), 1)
    recalls.append(recall)

plt.figure()
plt.plot(thresholds, recalls)
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.title("Sensitivity vs Threshold")
plt.show()

print("✅ All graphs generated successfully.")