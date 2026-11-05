import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Default Classifier", layout="wide")

# Header Banner using triple single quotes inside a double-quoted block
st.markdown('''
<div style='background-color:#0e1117; padding:20px; border-radius:10px; margin-bottom:25px;'>
    <h1 style='color:#ffffff; text-align:center; margin:0;'>Credit Card Default Classification Dashboard</h1>
    <p style='color:#a3a8b4; text-align:center; margin:5px 0 0 0;'>Interactive real-time prediction & model comparison platform</p>
</div>
''', unsafe_allow_html=True)

# Detect and locate the model folder autonomously
current_dir = os.getcwd()
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

# Model and Scaler loader
def load_artifacts_raw(model_dir):
    models = {}
    scaler = None
    if not model_dir:
        return models, None

    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Auto-discover all joblib files
    for f in os.listdir(model_dir):
        if f.endswith('.joblib') and f != 'scaler.joblib':
            model_name = f.replace('.joblib', '').replace('_', ' ').title()
            path = os.path.join(model_dir, f)
            try:
                models[model_name] = joblib.load(path)
            except:
                pass
    return models, scaler

models, scaler = load_artifacts_raw(found_model_dir)

# Main Columns
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### Configuration & Upload")

    if not models:
        st.error("No models loaded. Please check model paths.")
        model_choice = "None Available"
    else:
        st.success(f"Successfully loaded {len(models)} models!")
        model_choice = st.selectbox("**Choose Classification Model**", list(models.keys()))

    uploaded_file = st.file_uploader("**Upload Test CSV Data**", type="csv")

    st.markdown("---")
    st.markdown("📝 **Quick Info**:
" 
                "To make predictions, upload a CSV dataset containing client credit history (matching features `X1` to `X23`).")

with col2:
    st.markdown("### Dashboard & Prediction Hub")

    # Tab layouts for Clean Segregation
    tab1, tab2 = st.tabs(["Model Evaluation Summary", "Predict Default Risk"])

    with tab1:
        st.markdown("#### Performance Comparison (UCI Credit Default Dataset)")
        # Create explicit dataframe display matching analysis
        comparison_data = {
            "Model": ["Decision Tree", "Support Vector Machine", "Logistic Regression", "K-Nearest Neighbors", "Naive Bayes (Gaussian)"],
            "Accuracy": ["81.63%", "81.60%", "80.77%", "79.28%", "75.25%"],
            "AUC Score": [0.7444, 0.7210, 0.7076, 0.7014, 0.7249],
            "Precision": ["65.47%", "66.52%", "68.68%", "54.87%", "45.15%"],
            "Recall": ["35.87%", "33.84%", "23.96%", "35.64%", "55.39%"],
            "F1 Score": ["46.35%", "44.86%", "35.53%", "43.22%", "49.75%"],
            "MCC Score": [0.3879, 0.3809, 0.3244, 0.3233, 0.3386]
        }
        df_metrics = pd.DataFrame(comparison_data)
        st.dataframe(df_metrics, use_container_width=True)
        st.markdown("**Key Insights:** Decision Tree (Max Depth = 6) provides the most balanced predictions with optimal AUC (0.744) and accuracy.")

    with tab2:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("File Uploaded Successfully!")

            st.markdown("**Input Preview:**")
            st.dataframe(data.head(5), use_container_width=True)

            if st.button("Run Predictions", type="primary"):
                if scaler and model_choice in models:
                    try:
                        features = data.copy()
                        if 'Y' in features.columns:
                            features = features.drop(columns=['Y'])

                        scaled_features = scaler.transform(features)
                        predictions = models[model_choice].predict(scaled_features)

                        res_df = data.copy()
                        res_df['Predicted_Default'] = predictions

                        st.markdown("---")
                        st.markdown("#### Prediction Results")
                        st.dataframe(res_df, use_container_width=True)

                        # Summary Stats
                        defaults_count = int(np.sum(predictions))
                        st.metric("Flagged Defaults Identified", f"{defaults_count} / {len(predictions)}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Selected model configuration or scaler is missing.")
        else:
            st.info("Please upload a test dataset (like `test_data.csv` found in this directory) to run predictions.")
