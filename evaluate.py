import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset from the data folder
df = pd.read_csv("data/deep_slice_dataset.csv")
df_clean = df.dropna(subset=["Use CaseType (Input 1)", "Slice_Type"]).copy()

feature_cols = [
    "Use CaseType (Input 1)",
    "LTE/5G UE Category (Input 2)",
    "Technology Supported (Input 3)",
    "GBR (Input 6)",
    "Packet Loss Rate (Reliability)",
    "Packet Delay Budget (Latency)",
]
target_col = "Slice_Type"

X = df_clean[feature_cols].astype(str)
y = df_clean[target_col].astype(str)

label_encoders = joblib.load("label_encoders.pkl")
target_le = joblib.load("target_le.pkl")

for col in X.columns:
  X[col] = label_encoders[col].transform(X[col])

y_encoded = target_le.transform(y)
_, X_test, _, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = joblib.load("slice_model.pkl")
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

import numpy as np

# Find the unique classes present in the current test set
unique_classes_in_test = np.unique(y_test)
active_class_names = [target_le.classes_[i] for i in unique_classes_in_test]

# Print Metrics safely
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=unique_classes_in_test,
        target_names=active_class_names,
        zero_division=0,
    )
)

# Generate and save Confusion Matrix chart
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_le.classes_,
    yticklabels=target_le.classes_,
)
plt.xlabel("Predicted Slice")
plt.ylabel("Actual Slice")
plt.title("5G Slice Selection Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
print("Confusion matrix saved as 'confusion_matrix.png'!")