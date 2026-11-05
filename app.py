import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Default Classifier", layout="wide")

st.title("💳 Credit Card Default Classification Dashboard")
st.write("This application allows you to evaluate risk predictions using 5 different ML models.")

# Map user-friendly names to filenames
MODEL_MAP = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "k-nearest_neighbors",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian",
    "Support Vector Machine": "support_vector_machine"
}

# Model loader with robust path checking
@st.cache_resource
def load_artifacts():
    models = {}
    # Look in current directory model folder, or project-folder subfolder
    possible_paths = ['.', './project-folder']
    
    # Try to find where scaler is to determine the correct path prefix
    found_prefix = '.'
    for prefix in possible_paths:
        test_path = os.path.join(prefix, 'model', 'scaler.joblib')
        if os.path.exists(test_path):
            found_prefix = prefix
            break

    # Load scaler
    scaler_path = os.path.join(found_prefix, 'model', 'scaler.joblib')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Load models
    for display_name, file_name in MODEL_MAP.items():
        path = os.path.join(found_prefix, 'model', f'{file_name}.joblib')
        if os.path.exists(path):
            models[display_name] = joblib.load(path)
            
    return models, scaler, found_prefix

models, scaler, path_prefix = load_artifacts()

# Sidebar options
st.sidebar.header("Configuration")

# Debugging path visualizer in sidebar to ensure files are visible
st.sidebar.write(f"Detected active folder: `{os.path.abspath(path_prefix)}`")

if not models:
    st.sidebar.error("❌ No models found! Please verify model files exist inside the `./model/` directory.")
else:
    st.sidebar.success(f"✅ Successfully loaded {len(models)} models!")

model_choice = st.sidebar.selectbox("Choose Classification Model", list(models.keys()) if models else ["No models loaded"])

uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Check if we should drop a target column
    features = data.copy()
    if 'Y' in features.columns:
        features = features.drop(columns=['Y'])

    if st.button("Run Predictions"):
        if scaler and model_choice in models:
            scaled_features = scaler.transform(features)
            predictions = models[model_choice].predict(scaled_features)

            res_df = data.copy()
            res_df['Predicted_Default'] = predictions

            st.subheader("Prediction Results")
            st.dataframe(res_df)
        else:
            st.error("Selected model or scaler artifacts could not be loaded successfully.")
else:
    st.info("Please upload a test dataset (like `test_data.csv` found in this repository) to run predictions.")
