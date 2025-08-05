import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
location_encoder = joblib.load("location_encoder.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #002B5B; font-size: 42px;'>
        🏠 Pune City House Price Predictor
    </h1>
    <p style='text-align: center; font-size: 18px;'>Built with ML by <b>Sushant Munde</b></p>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

st.write("Fill the details below to estimate the house price:")

location = st.selectbox("Select Location", location_encoder.classes_)
area = st.number_input("Area (in sqft)", min_value=200, max_value=10000, step=50)
bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, step=1)

if st.button("Predict Price"):
    loc_encoded = location_encoder.transform([location])[0]
    input_data = np.array([[area, bhk, bathrooms, loc_encoded]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"🏷️ Estimated Price: ₹ {predicted_price:.2f} lakhs")
