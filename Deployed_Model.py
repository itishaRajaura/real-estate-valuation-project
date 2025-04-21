import streamlit as st
import pandas as pd
import joblib
import sklearn

st.set_page_config(page_title="Real Estate Price Estimator", page_icon="🏘️")

st.title("🏘️ Real Estate Price Estimator")
st.write("Estimate the *house price per unit area* based on location and features.")

# Try loading the model with error handling
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.joblib")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# Sidebar inputs
st.sidebar.header("Property Features")
house_age = st.sidebar.slider("House Age (years)", 0.0, 50.0, 10.0)
dist_to_mrt = st.sidebar.slider("Distance to Nearest MRT (meters)", 0.0, 6500.0, 1000.0)
n_convenience = st.sidebar.number_input("Number of Convenience Stores Nearby", 0, 20, 5)
latitude = st.sidebar.number_input("Latitude", 24.90, 25.10, 24.98)
longitude = st.sidebar.number_input("Longitude", 121.40, 121.60, 121.50)
trans_year = st.sidebar.slider("Transaction Year", 2012, 2013, 2013)
trans_month = st.sidebar.slider("Transaction Month", 1, 12, 6)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([{
        "house_age": house_age,
        "dist_to_mrt": dist_to_mrt,
        "n_convenience": n_convenience,
        "latitude": latitude,
        "longitude": longitude,
        "trans_year": trans_year,
        "trans_month": trans_month
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.subheader("💰 Predicted Price per Unit Area:")
        st.success(f"{prediction:.2f} currency units")
        st.markdown("📌 Note: Based on historical Taipei real estate data")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
