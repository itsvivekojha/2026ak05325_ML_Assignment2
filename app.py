import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Default Classifier", layout="wide")

st.title("💳 Credit Card Default Classification Dashboard")
st.write("This application allows you to evaluate risk predictions using 5 different ML models.")

# Model loader
@st.cache_resource
def load_artifacts():
    models = {}
    model_names = ['logistic_regression', 'decision_tree', 'k-nearest_neighbors', 'naive_bayes_gaussian', 'support_vector_machine']
    for name in model_names:
        path = f'project-folder/model/{name}.joblib'
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load('project-folder/model/scaler.joblib') if os.path.exists('project-folder/model/scaler.joblib') else None
    return models, scaler

models, scaler = load_artifacts()

# Sidebar options
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose Classification Model", list(models.keys()))

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
            st.error("Model artifacts could not be loaded successfully.")
else:
    st.info("Please upload a test dataset (like `test_data.csv` found in this repository) to run predictions.")
