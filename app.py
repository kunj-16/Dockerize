
import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load("model.pkl")

# Configure Streamlit page
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Application heading
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger details below to predict "
    "whether the passenger would have survived."
)

st.divider()

# Passenger information
st.subheader("Passenger Information")

# Input fields
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

# Prediction
if st.button("Predict Survival"):

    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    prediction = model.predict(input_data)

    st.divider()

    if prediction[0] == 1:
        st.success("Prediction: Passenger would have survived! 🎉")
    else:
        st.error("Prediction: Passenger would not have survived.")
