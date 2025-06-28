import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading dataset 
diabetes_dataset = pd.read_csv("diabetes.csv")
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split and train for testing purpose
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# frontend portion using streamlit
st.set_page_config(layout="wide")
st.title("🧬 Diabetes Prediction App")
st.write("Please enter the details below:")

# creating a 4 col for better ui
col1, col2, col3, col4 = st.columns(4)

with col1:
    Pregnancies = st.text_input("Pregnancies", value="0")
    SkinThickness = st.text_input("Skin Thickness", value="20")

with col2:
    Glucose = st.text_input("Glucose Level", value="120")
    Insulin = st.text_input("Insulin", value="80")

with col3:
    BloodPressure = st.text_input("Blood Pressure", value="70")
    BMI = st.text_input("BMI", value="25.0")

with col4:
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", value="0.5")
    Age = st.text_input("Age", value="30")

if st.button("Predict Diabetes"):
    try:
        input_data = (
            float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
            float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
        )
        input_data_np = np.asarray(input_data).reshape(1, -1)
        std_data = scaler.transform(input_data_np)
        prediction = classifier.predict(std_data)

        if prediction[0] == 0:
            st.success("✅ The person is **not diabetic**.")
        else:
            st.error("🚨 The person **is diabetic**.")
    except ValueError:
        st.warning("⚠️ Please enter valid numerical values in all fields.")

# Sidebar
st.sidebar.title("📊 Model Accuracy")
st.sidebar.write("**Training Accuracy:**", round(accuracy_score(Y_train, classifier.predict(X_train)) * 100, 2), "%")
st.sidebar.write("**Test Accuracy:**", round(accuracy_score(Y_test, classifier.predict(X_test)) * 100, 2), "%")
