import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load dataset and slice only the clean training segment (Rows 0 to 24)
df = pd.read_csv("data/deep_slice_dataset.csv")
df_clean = df.iloc[0:25].dropna(subset=["Use CaseType (Input 1)"]).copy()

# Select input features and target
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

# Encode categorical features properly
label_encoders = {}
for col in X.columns:
  le = LabelEncoder()
  X[col] = le.fit_transform(X[col])
  label_encoders[col] = le

# Encode target labels
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(clf, "slice_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_le, "target_le.pkl")
print("Model re-trained cleanly using valid dataset rows!")

# Save feature names for visualization
joblib.dump(feature_cols, "feature_cols.pkl")