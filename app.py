import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------------------
# Load model and scaler
# ---------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)



# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to check the risk of heart disease. "
         "The app uses a Machine Learning model trained on medical data.")

# ---------------------------
# User Inputs
# ---------------------------

age = st.number_input("Age of the Patient", 18, 100, 45)

sex = st.radio("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina (chest pain related to heart)",
     "Atypical Angina (pain not directly heart-related)",
     "Non-anginal Pain (not heart related)",
     "Asymptomatic (no pain but disease present)"]
)
cp = ["Typical Angina (chest pain related to heart)",
      "Atypical Angina (pain not directly heart-related)",
      "Non-anginal Pain (not heart related)",
      "Asymptomatic (no pain but disease present)"].index(cp)

trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox(
    "Resting ECG Result",
    ["Normal",
     "ST-T wave abnormality",
     "Left Ventricular Hypertrophy"]
)
restecg = ["Normal",
           "ST-T wave abnormality",
           "Left Ventricular Hypertrophy"].index(restecg)

thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exang = st.radio("Exercise Induced Angina?", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping (healthy)",
     "Flat (possible heart disease)",
     "Downsloping (high risk)"]
)
slope = ["Upsloping (healthy)",
         "Flat (possible heart disease)",
         "Downsloping (high risk)"].index(slope)

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

thal = st.selectbox(
    "Thalassemia Result",
    ["Normal",
     "Fixed Defect (permanent damage)",
     "Reversible Defect (damage under stress)"]
)
thal = ["Normal",
        "Fixed Defect (permanent damage)",
        "Reversible Defect (damage under stress)"].index(thal) + 1  # dataset uses 1,2,3

# ---------------------------
# Prediction
# ---------------------------

feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

input_data = pd.DataFrame(
    [[age, sex, cp, trestbps, chol, fbs,
      restecg, thalach, exang, oldpeak,
      slope, ca, thal]],
    columns=feature_names
)

input_data_scaled = input_data

if st.button("🔍 Predict"):

    prediction = model.predict(input_data_scaled)[0]

    probabilities = model.predict_proba(input_data_scaled)[0]

    classes = list(model.classes_)

    disease_probability = probabilities[classes.index(1)]
    no_disease_probability = probabilities[classes.index(0)]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ High risk of Heart Disease "
            f"(Model Probability: {disease_probability*100:.2f}%)"
        )
    else:
        st.success(
            f"✅ Low risk of Heart Disease "
            f"(Model Probability: {no_disease_probability*100:.2f}%)"
        )

    st.write("### Probability Breakdown")

    st.write(
        f"No Heart Disease: {no_disease_probability*100:.2f}%"
    )

    st.write(
        f"Heart Disease: {disease_probability*100:.2f}%"
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Machine Learning (Logistic Regression, Random Forest, XGBoost).")
