'''
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open('models.pkl', 'rb'))
if isinstance(model, list):
    model = model[0]

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #FF4B2B;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)

# Two-column user input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('🎂 Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('👤 Sex', ['Male', 'Female'])
    cp = st.selectbox('💢 Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('🩺 Resting Blood Pressure (mm Hg)', min_value=70, max_value=250, value=120)
    chol = st.number_input('🍳 Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('🍬 Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])

with col2:
    restecg = st.selectbox('🧠 Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.number_input('🏃 Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('🏋️ Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('📉 ST Depression (oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox('📈 Slope of ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.number_input('🩸 Number of Major Vessels (ca)', min_value=0, max_value=4, value=0)
    thal = st.selectbox('🧬 Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Encoding inputs
try:
    sex_val = 1 if sex == 'Male' else 0
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_val = 1 if fbs == 'True' else 0
    restecg_map = {'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_val = 1 if exang == 'Yes' else 0
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    input_data = np.array([[
        age,
        sex_val,
        cp_map[cp],
        trestbps,
        chol,
        fbs_val,
        restecg_map[restecg],
        thalach,
        exang_val,
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal]
    ]])

    # Prediction
    if st.button("🔍 Predict"):
        result = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1] * 100 if hasattr(model, 'predict_proba') else None

        st.markdown("---")
        if result == 1:
            st.error("🚨 High Risk of Heart Disease Detected")
            if proba:
                st.markdown(f"🔴 Prediction Confidence: `{proba:.2f}%`")
        else:
            st.success("✅ No Heart Disease Detected")
            if proba:
                st.markdown(f"🟢 Prediction Confidence: `{proba:.2f}%`")

except Exception as e:
    st.warning(f"⚠️ Input Error: {e}")

'''
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('models.pkl', 'rb'))
if isinstance(model, list):
    model = model[0]

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# 🔥 Custom style
st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: #FF4B2B;
        }
        div.stButton > button {
            background: linear-gradient(to right, #FF416C, #FF4B2B);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 30px;
            border: none;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }
        div.stButton > button:hover {
            background: linear-gradient(to right, #FF4B2B, #FF416C);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1>💓 Heart Disease Prediction</h1>", unsafe_allow_html=True)

# Input form layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('🎂 Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('👤 Sex', ['Male', 'Female'])
    cp = st.selectbox('💢 Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('🩺 Resting Blood Pressure (mm Hg)', min_value=70, max_value=250, value=120)
    chol = st.number_input('🍳 Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('🍬 Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])

with col2:
    restecg = st.selectbox('🧠 Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.number_input('🏃 Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('🏋️ Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('📉 ST Depression (oldpeak)', min_value=0, max_value=6, value=1, step=1)
    slope = st.selectbox('📈 Slope of ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.number_input('🩸 Number of Major Vessels (ca)', min_value=0, max_value=4, value=0)
    thal = st.selectbox('🧬 Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Encode categorical variables
try:
    sex_val = 1 if sex == 'Male' else 0
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_val = 1 if fbs == 'True' else 0
    restecg_map = {'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_val = 1 if exang == 'Yes' else 0
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    input_data = np.array([[
        age,
        sex_val,
        cp_map[cp],
        trestbps,
        chol,
        fbs_val,
        restecg_map[restecg],
        thalach,
        exang_val,
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal]
    ]])

    # Left-aligned Predict button
    if st.button("🔍 Predict"):
        result = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1] * 100 if hasattr(model, 'predict_proba') else None

        st.markdown("---")
        if result == 1:
            st.error("🚨 High Risk of Heart Disease Detected")
            if proba:
                st.markdown(f"🔴 Prediction Confidence: `{proba:.2f}%`")
        else:
            st.success("✅ No Heart Disease Detected")
            if proba:
                st.markdown(f"🟢 Prediction Confidence: `{proba:.2f}%`")

except Exception as e:
    st.warning(f"⚠️ Input Error: {e}")
