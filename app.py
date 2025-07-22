
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("clean_salary_model.pkl")

# Page configuration
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# Title and Description
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar Inputs (numeric - already label encoded or continuous)
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.slider("Education Number (1–16)", 1, 16, 13)
occupation_code = st.sidebar.number_input("Occupation Code (0–13)", min_value=0, max_value=13, value=4)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
workclass = st.sidebar.slider("Workclass", 0, 40, 5)

# Input DataFrame (matches training data structure)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation_code],
    'hours-per-week': [hours_per_week],
    'workclass': [workclass]
})

# Show input data
st.write("### 🔎 Input Data")
st.write(input_df)

# Predict Button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    label = "Salary > 50K" if prediction[0] == 1 else "Salary ≤ 50K"
    st.success(f"✅ Prediction: {label}")
 

# Batch Prediction Section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    
    # Download button
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
