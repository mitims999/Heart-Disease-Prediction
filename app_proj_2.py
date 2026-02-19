import streamlit as st
import  pandas as pd
import joblib 

model = joblib.load("LogisticRegression_heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
expected_cols = joblib.load("columns_heart.pkl")


st.title("heart stroke prediction")
st.markdown("provide the following details")

age = st.slider("Age:", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
Chest_pain = st.selectbox("chest pain type", ["ATA", "NAP", "TA", "ASY"])
RESTING_BP = st.number_input("resting blood pressure (mm hg)", 80, 200, 120)
cholesterol = st.number_input("cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("fasting blood sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("resting ecg", ["Normal", "ST", "LVH"])
max_hr = st.slider("max heart rate", 60, 220, 150)
excercise_angina = st.selectbox("excercise induced angina", ["Y", "N"])
oldpeak = st.slider("oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': RESTING_BP,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + Chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + excercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("high risk of heart disease")
    else:
        st.success("low risk")