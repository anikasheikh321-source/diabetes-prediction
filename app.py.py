import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction")
st.title("Diabetes Prediction")
st.write("Enter the feature values below to predict diabetes.")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0.0, format="%.4f")
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, format="%.4f")
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, format="%.4f")
Insulin = st.number_input("Insulin", min_value=0.0, format="%.4f")
BMI = st.number_input("BMI", min_value=0.0, format="%.4f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.4f")
Age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict Diagnosis"):
    input_data = np.array([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("Diagnosis: Diabetic")
    else:
        st.success("Diagnosis: Not Diabetic")
