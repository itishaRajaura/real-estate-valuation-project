import streamlit as st
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# Page setup
st.set_page_config(page_title="Real Estate Valuation", layout="centered")
st.title("🏠 Real Estate Price Prediction")

st.markdown("""
Enter the property details below to estimate the *price per unit area*.
""")

# Input form
with st.form("input_form"):
    house_age = st.number_input("House Age (years)", min_value=0.0, max_value=100.0, step=0.1)
    dist_to_mrt = st.number_input("Distance to Nearest MRT Station (meters)", min_value=0.0, step=1.0)
    n_convenience = st.number_input("Number of Convenience Stores Nearby", min_value=0, step=1)
    latitude = st.number_input("Latitude", min_value=0.0, step=0.0001)
    longitude = st.number_input("Longitude", min_value=0.0, step=0.0001)
    trans_year = st.selectbox("Transaction Year", options=[2012, 2013, 2014])
    trans_month = st.selectbox("Transaction Month", options=list(range(1, 13)))

    submitted = st.form_submit_button("Predict")

# When form is submitted
if submitted:
    input_data = np.array([[house_age, dist_to_mrt, n_convenience,
                            latitude, longitude, trans_year, trans_month]])
    prediction = model.predict(input_data)[0]
    st.success(f"🏷️ *Estimated Price per Unit Area: {prediction:.2f}*")
