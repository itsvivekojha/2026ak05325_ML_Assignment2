# Credit Card Default Classification System

This repository provides a comparison of 5 machine learning models evaluating predictions on the Credit Card Default Dataset from UCI. It includes a Streamlit application to interface and test predictions in real-time.

## Repository Layout
- `app.py`: The entrypoint dashboard of our interactive application.
- `requirements.txt`: Necessary dependencies required to run the workspace.
- `test_data.csv`: A sample test set of the features to trial evaluation.
- `model/`: Directory storing binary files of our trained models and scaler.

## How to Run locally:
1. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
