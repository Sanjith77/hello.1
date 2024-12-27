import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title of the app
st.title("Bankruptcy Prevention App")

# Step 1: Load the trained model
model_file = "Bankruptcy_.pkl"

try:
    model = joblib.load(model_file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file '{model_file}' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Step 2: Define the expected features
expected_features = [
    "industrial_risk", "management_risk", "financial_flexibility",
    "credibility", "competitiveness", "operating_risk",
    "feature_7", "feature_8", "feature_9", "feature_10", "feature_11"
]

# Step 3: Upload the dataset
uploaded_file = st.file_uploader("Upload the dataset file (Excel format)", type=["xlsx"])

if uploaded_file:
    try:
        # Load the dataset
        data = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Display available columns
        st.write("Available columns in the dataset:")
        st.write(data.columns.tolist())

        # Check for missing features and add them with default values
        missing_features = [feature for feature in expected_features if feature not in data.columns]
        for feature in missing_features:
            data[feature] = 0  # Default value can be adjusted as needed

        # Prepare input features for prediction
        X = data[expected_features]

        st.write("Enter input values to predict bankruptcy:")
        user_input = []
        for column in expected_features:
            value = st.number_input(f"Enter value for {column}:", value=0.0)
            user_input.append(value)

        # Predict based on user input
        if st.button("Predict"):
            user_input = np.array(user_input).reshape(1, -1)
            prediction = model.predict(user_input)

            # Map prediction to meaningful output
            prediction_text = "Non-Bankruptcy" if prediction[0] == 1 else "Bankruptcy"
            st.write(f"Prediction: {prediction_text}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a file to proceed.")
