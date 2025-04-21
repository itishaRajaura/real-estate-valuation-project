import streamlit as st
import pandas as pd
import joblib

# Load trained model
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Model file not found! Make sure 'best_model.joblib' is in the same directory.")
    st.stop()  # Stop the app if model file is not found

st.title("🏘️ Real Estate Price Estimator")
st.write("Estimate the **house price per unit area** based on location and features.")

# Sidebar input fields
st.sidebar.header("Property Features")

house_age = st.sidebar.slider("House Age (years)", 0.0, 50.0, 10.0)
dist_to_mrt = st.sidebar.slider("Distance to Nearest MRT (meters)", 0.0, 6500.0, 1000.0)
n_convenience = st.sidebar.number_input("Number of Convenience Stores Nearby", 0, 20, 5)
latitude = st.sidebar.number_input("Latitude", 24.90, 25.10, 24.98)
longitude = st.sidebar.number_input("Longitude", 121.40, 121.60, 121.50)
trans_year = st.sidebar.slider("Transaction Year", 2012, 2013, 2013)
trans_month = st.sidebar.slider("Transaction Month", 1, 12, 6)

# Predict button
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([{
        "house_age": house_age,
        "dist_to_mrt": dist_to_mrt,
        "n_convenience": n_convenience,
        "latitude": latitude,
        "longitude": longitude,
        "trans_year": trans_year,
        "trans_month": trans_month
    }])

    # Make the prediction
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Price per Unit Area:")
    st.success(f"{prediction:.2f} (currency units)")

    st.markdown("📌 Note: This is an estimate based on historical data from Taipei housing market.")
