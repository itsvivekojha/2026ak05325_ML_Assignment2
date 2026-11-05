import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Default Classifier", layout="wide")

st.title("💳 Credit Card Default Classification Dashboard")

# Diagnostic Section
st.subheader("📂 App Environment & Path Diagnostics")
current_dir = os.getcwd()
st.write(f"**Current Working Directory:** `{current_dir}`")

# Resolve absolute paths to prevent any lookup ambiguity
possible_folders = [
    current_dir,
    os.path.join(current_dir, 'project-folder'),
    os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else current_dir
]

found_model_dir = None
for folder in possible_folders:
    test_path = os.path.join(folder, 'model')
    if os.path.exists(test_path):
        found_model_dir = os.path.abspath(test_path)
        break

if found_model_dir:
    st.success(f"✅ Found model directory at: `{found_model_dir}`")
    available_files = os.listdir(found_model_dir)
    st.write(f"**Files inside model directory:** `{available_files}`")
else:
    st.error("❌ Could not locate the `model` folder. Please ensure the `model/` folder containing `.joblib` files is in the same directory as this script.")

# Raw model and Scaler loader (without st.cache_resource to prevent caching issues)
def load_artifacts_raw(model_dir):
    models = {}
    scaler = None
    if not model_dir:
        return models, None

    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Auto-discover all files ending in .joblib except the scaler
    for f in os.listdir(model_dir):
        if f.endswith('.joblib') and f != 'scaler.joblib':
            model_name = f.replace('.joblib', '').replace('_', ' ').title()
            path = os.path.join(model_dir, f)
            try:
                models[model_name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {f}: {e}")

    return models, scaler

models, scaler = load_artifacts_raw(found_model_dir)

# Sidebar options
st.sidebar.header("Configuration")

if not models:
    st.sidebar.error("❌ No models loaded.")
    model_choice = "None Available"
else:
    st.sidebar.success(f"✅ Loaded {len(models)} models!")
    model_choice = st.sidebar.selectbox("Choose Classification Model", list(models.keys()))

uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Check if we should drop target column Y
    features = data.copy()
    if 'Y' in features.columns:
        features = features.drop(columns=['Y'])

    if st.button("Run Predictions"):
        if scaler and model_choice in models:
            try:
                scaled_features = scaler.transform(features)
                predictions = models[model_choice].predict(scaled_features)

                res_df = data.copy()
                res_df['Predicted_Default'] = predictions

                st.subheader("Prediction Results")
                st.dataframe(res_df)
            except Exception as e:
                st.error(f"Prediction execution failed: {e}")
        else:
            st.error("Selected model or scaler is missing. Verify the path diagnostic output above.")
else:
    st.info("Please upload a test dataset (like `test_data.csv` found in this repository) to run predictions.")
