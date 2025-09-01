import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS for background and input/label styling
st.markdown(
    """
    <style>
    /* Main page background */
    .stApp {
        background-color: #2b2b2b;  /* lighter dark gray */
        color: #f0f0f0;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00ffff;
    }
    /* Input boxes */
    .stNumberInput>div>input {
        background-color: #3a3a3a;
        color: #ffffff;
        font-weight: bold;
    }
    /* Input labels (V1, V2, Amount) */
    .stNumberInput>label>div {
        color: #ffcc00;  /* bright yellow */
        font-weight: bold;
    }
    /* Button */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud.")

# Input fields
V1 = st.number_input("V1")
V2 = st.number_input("V2")
Amount = st.number_input("Amount")

if st.button("Predict"):
    input_data = np.array([[V1, V2, Amount]], dtype=float)
    input_data[:, 2] = scaler.transform(input_data[:, 2].reshape(-1,1)).flatten()
    pred = model.predict(input_data)
    st.markdown(f"<h2>Prediction: {'Fraud ❌' if pred[0]==1 else 'Legit ✅'}</h2>", unsafe_allow_html=True)
