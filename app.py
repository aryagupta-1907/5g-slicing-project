import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved assets
model = joblib.load("slice_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_le = joblib.load("target_le.pkl")
feature_cols = joblib.load("feature_cols.pkl")

st.title("5G Intelligent Network Slice Selection Dashboard")
st.write(
    "Dynamically maps device requirements to **eMBB**, **URLLC**, or **mMTC**"
    " network slices using Machine Learning & 5G Core NSSF simulation."
)

# Sidebar navigation
option = st.sidebar.selectbox(
    "Navigation",
    [
        "Single Device Selector",
        "Batch CSV Prediction",
        "Model Performance",
        "Dataset Analytics",
    ],
)

if option == "Single Device Selector":
  st.header("Single Device 5G Slicing")

  use_case = st.selectbox(
      "Use Case Type", label_encoders["Use CaseType (Input 1)"].classes_
  )
  ue_category = st.selectbox(
      "LTE/5G UE Category", label_encoders["LTE/5G UE Category (Input 2)"].classes_
  )
  tech_supported = st.selectbox(
      "Technology Supported",
      label_encoders["Technology Supported (Input 3)"].classes_,
  )
  gbr = st.selectbox("GBR Status", label_encoders["GBR (Input 6)"].classes_)
  plr = st.selectbox(
      "Packet Loss Rate (Reliability)",
      label_encoders["Packet Loss Rate (Reliability)"].classes_,
  )
  pdb = st.selectbox(
      "Packet Delay Budget (Latency)",
      label_encoders["Packet Delay Budget (Latency)"].classes_,
  )

  if st.button("Assign Network Slice"):
    input_data = pd.DataFrame(
        [[use_case, ue_category, tech_supported, gbr, plr, pdb]],
        columns=feature_cols,
    )

    for col in input_data.columns:
      input_data[col] = label_encoders[col].transform(input_data[col])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    assigned_slice = target_le.inverse_transform(prediction)[0]

    st.success(
        f"Optimal Network Slice Assigned: **{assigned_slice.upper()}**"
    )

    st.write("### 5G Core (NSSF) Policy JSON Payload:")
    st.json({
        "statusCode": 200,
        "message": "Slice Allocation Successful via ML Policy",
        "sliceType": assigned_slice,
        "isolationLevel": "Dedicated Logical Subnet",
    })

elif option == "Batch CSV Prediction":
  st.header("Batch Device Slicing (CSV Upload)")
  uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
  if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", batch_df.head())

    if st.button("Run Batch Slicing"):
      X_batch = batch_df[feature_cols].astype(str)
      for col in X_batch.columns:
        X_batch[col] = label_encoders[col].transform(X_batch[col])

      X_batch_scaled = scaler.transform(X_batch)
      batch_preds = model.predict(X_batch_scaled)
      batch_df["Assigned_Slice"] = target_le.inverse_transform(batch_preds)

      st.write("### Slicing Results:", batch_df)
      st.download_button(
          "Download Results CSV",
          batch_df.to_csv(index=False).encode("utf-8"),
          "batch_slice_results.csv",
          "text/csv",
      )

elif option == "Model Performance":
  st.header("Machine Learning Evaluation & Metrics")
  if os.path.exists("confusion_matrix.png"):
    st.image(
        "confusion_matrix.png",
        caption="Confusion Matrix across eMBB, URLLC, and mMTC slices",
    )
  else:
    st.warning("Run `python evaluate.py` first to generate the confusion matrix!")

  # Feature Importance Visualization
  st.subheader("Feature Importance (Explainable AI)")
  importances = model.feature_importances_
  fig, ax = plt.subplots(figsize=(8, 4))
  sns.barplot(x=importances, y=feature_cols, ax=ax, palette="viridis")
  ax.set_title("Which Parameters Drive 5G Slice Selection?")
  ax.set_xlabel("Importance Score")
  ax.set_ylabel("Network Feature")
  st.pyplot(fig)

elif option == "Dataset Analytics":
  st.header("DeepSlice Dataset Overview")
  df = pd.read_csv("data/deep_slice_dataset.csv")
  df_clean = df.iloc[0:25].dropna(subset=["Use CaseType (Input 1)"])

  st.write("Distribution of Target Slices in Training Data:")
  slice_counts = df_clean["Slice_Type"].value_counts()

  fig, ax = plt.subplots(figsize=(6, 4))
  sns.barplot(
      x=slice_counts.index,
      y=slice_counts.values,
      ax=ax,
      palette=["#ff9999", "#66b3ff", "#99ff99"],
  )
  ax.set_title("Training Samples per Network Slice")
  ax.set_xlabel("Slice Type")
  ax.set_ylabel("Count")
  st.pyplot(fig)