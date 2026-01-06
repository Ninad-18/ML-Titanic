import streamlit as st
import numpy as np
import joblib
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()
model_path = os.getenv("MODEL_PATH")

# =========================
# LOAD MODEL 
# =========================
model = joblib.load(model_path)


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability.")

# =========================
# USER INPUTS
# =========================
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings / Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents / Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# =========================
# ENCODING USER INPUT
# =========================
sex_encoded = 1 if sex == "male" else 0

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# =========================
# CREATE INPUT ARRAY
# =========================
input_data = np.array([[
    pclass,
    age,
    sibsp,
    parch,
    fare,
    sex_encoded,
    embarked_Q,
    embarked_S
]])


# =========================
# PREDICTION
# =========================
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # convert to %

    if prediction == 1:
        st.success(
            f"🎉 Passenger Survived! (Probability: {probability:.2f}%)"
        )
    else:
        st.error(
            f"❌ Passenger Did Not Survive (Probability: {probability:.2f}%)"
        )
