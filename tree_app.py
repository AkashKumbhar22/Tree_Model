import streamlit as st
import pandas as pd
import joblib

# Load model and label encoder
try:
    rf_model = joblib.load("tree_health_model.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or label encoder: {e}")
    st.stop()

# Sample credentials (you can replace this with DB/API in production)
users = {
    "admin": "password123",
    "akash": "treehealth2025"
}

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Login Page ---
def login_page():
    st.title("🔐 Login to Tree Health App")

    username = st.text_input("Username (max 20 chars)").strip()
    password = st.text_input("Password (max 20 chars)", type="password").strip()

    if st.button("Login"):
        if len(username) > 20:
            st.warning("❗ Username must be at most 20 characters.")
        elif len(password) > 20:
            st.warning("❗ Password must be at most 20 characters.")
        elif username in users and users[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password.")

# --- Prediction Page ---
def prediction_page():
    st.title("🌳 Tree Health Prediction App")
    st.subheader(f"👤 Logged in as: `{st.session_state.username}`")

    if st.button("Logout 🔓"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.experimental_rerun()

    st.header("Enter Tree Details")

    Latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0)
    Longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0)
    DBH = st.number_input("Diameter at Breast Height (DBH) [cm]", min_value=1.0, max_value=300.0)
    Tree_Height = st.number_input("Tree Height [m]", min_value=1.0, max_value=130.0)
    Crown_Width_North_South = st.number_input("Crown Width North-South [m]", min_value=0.5, max_value=60.0)
    Crown_Width_East_West = st.number_input("Crown Width East-West [m]", min_value=0.5, max_value=60.0)
    Slope = st.number_input("Slope [°]", min_value=0.0, max_value=90.0)
    Elevation = st.number_input("Elevation [m]", min_value=-430.0, max_value=8848.0)
    Temperature = st.number_input("Temperature [°C]", min_value=-50.0, max_value=60.0)
    Humidity = st.number_input("Humidity [%]", min_value=0.0, max_value=100.0)
    Soil_TN = st.number_input("Soil Total Nitrogen (TN) [%]", min_value=0.01, max_value=2.0)
    Soil_TP = st.number_input("Soil Total Phosphorus (TP) [%]", min_value=0.01, max_value=1.0)
    Soil_AP = st.number_input("Soil Available Phosphorus (AP) [%]", min_value=0.001, max_value=0.6)
    Soil_AN = st.number_input("Soil Available Nitrogen (AN) [%]", min_value=0.001, max_value=1.5)
    Menhinick_Index = st.number_input("Menhinick Index", min_value=0.0, max_value=10.0)
    Gleason_Index = st.number_input("Gleason Index", min_value=0.0, max_value=20.0)
    Disturbance_Level = st.number_input("Disturbance Level", min_value=0.0, max_value=1.0)
    Fire_Risk_Index = st.number_input("Fire Risk Index", min_value=0.0, max_value=1.0)

    if st.button("Predict Tree Health"):
        input_data = {
            'Latitude': Latitude,
            'Longitude': Longitude,
            'DBH': DBH,
            'Tree_Height': Tree_Height,
            'Crown_Width_North_South': Crown_Width_North_South,
            'Crown_Width_East_West': Crown_Width_East_West,
            'Slope': Slope,
            'Elevation': Elevation,
            'Temperature': Temperature,
            'Humidity': Humidity,
            'Soil_TN': Soil_TN,
            'Soil_TP': Soil_TP,
            'Soil_AP': Soil_AP,
            'Soil_AN': Soil_AN,
            'Menhinick_Index': Menhinick_Index,
            'Gleason_Index': Gleason_Index,
            'Disturbance_Level': Disturbance_Level,
            'Fire_Risk_Index': Fire_Risk_Index
        }

        input_df = pd.DataFrame([input_data])

        try:
            prediction_encoded = rf_model.predict(input_df)[0]
            prediction_label = le.inverse_transform([prediction_encoded])[0]
            st.success(f"🌿 Predicted Health Status: **{prediction_label}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# --- Main Control ---
if st.session_state.authenticated:
    prediction_page()
else:
    login_page()
