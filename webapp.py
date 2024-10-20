import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the best model from the pickle file
best_model = load('best_model.pkl')

# Title of the web app
st.title("Machine Failure Prediction")

# Create input fields for each feature based on your dataset
st.header("Input Features")

# Input fields for numerical features with default values
air_temperature = st.number_input("Air Temperature (K)", min_value=0.0, value=300.7)
process_temperature = st.number_input("Process Temperature (K)", min_value=0.0, value=311.9)
rotational_speed = st.number_input("Rotational Speed (rpm)", min_value=0.0, value=float(1366))  # Convert to float
torque = st.number_input("Torque (Nm)", min_value=0.0, value=float(55.1))  # Convert to float
tool_wear = st.number_input("Tool Wear (min)", min_value=0.0, value=float(36))  # Convert to float

# Input field for categorical feature with a default option
type_of_machine = st.selectbox("Type of Machine", options=["L", "M", "H"], index=0)  # Adjust index based on actual options

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Air_temperature__K_': [air_temperature],
    'Process_temperature__K_': [process_temperature],
    'Rotational_speed__rpm_': [rotational_speed],
    'Torque__Nm_': [torque],
    'Tool_wear__min_': [tool_wear],
    'Type': [type_of_machine]
})

# Prediction button
if st.button("Predict"):
    # Make prediction
    prediction = best_model.predict(input_data)
    prediction_proba = best_model.predict_proba(input_data)[:, 1]  # Get probability for class 1

    # Display the results
    st.success(f"Prediction: {'Machine Failure' if prediction[0] == 1 else 'No Machine Failure'}")
    st.write(f"Probability of Machine Failure: {prediction_proba[0]:.2f}")
