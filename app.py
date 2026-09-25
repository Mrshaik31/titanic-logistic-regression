import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Encoder
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("titanic_logistic_model1.pkl")
    deck_encoder = joblib.load("deck_encoder.pkl")
    return model, deck_encoder


model, deck_encoder = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter the passenger details below to predict "
    "the probability of survival."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0,
    step=1.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["C", "Q", "S"]
)

deck = st.selectbox(
    "Deck",
    ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]
)


# --------------------------------------------------
# Convert Categorical Values
# --------------------------------------------------

# Training encoding:
# Male = 0
# Female = 1

sex_value = 1 if sex == "Female" else 0


# Training encoding:
# C = 0
# Q = 1
# S = 2

embarked_mapping = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked_value = embarked_mapping[embarked]


# --------------------------------------------------
# Encode Deck
# --------------------------------------------------

try:
    deck_value = deck_encoder.transform([deck])[0]

except ValueError:
    st.error("Invalid deck value.")
    st.stop()


# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_value],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_value],
    "Deck": [deck_value]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Survival"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ The model predicts: Survived")
    else:
        st.error("❌ The model predicts: Did Not Survive")

    st.info(
        f"Survival Probability: **{probability * 100:.2f}%**"
    )


# --------------------------------------------------
# Optional: Show Input Data
# --------------------------------------------------

with st.expander("View Encoded Input Data"):
    st.dataframe(input_data)