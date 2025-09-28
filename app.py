
import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('house_price_model.pkl', 'rb'))

st.title("🏠 House Price Predictor")

area = st.number_input("Enter Area (sqft):", min_value=100, step=50)
bhk = st.number_input("Enter BHK:", min_value=1, step=1)

if st.button("Predict Price"):
    features = np.array([[area, bhk]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Price: {prediction:.2f} Lakhs")
