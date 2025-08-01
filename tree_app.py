import streamlit as st
import pandas as pd
import joblib
import requests
import json

# ------------------------------
# 1. Gemini API Configuration
# ------------------------------
GEMINI_API_KEY = "AIzaSyBWYikFZtblzgel33kCbzE0Ibl1qjt887o"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ------------------------------
# 2. Gemini API Call
# ------------------------------
@st.cache_data(show_spinner="Thinking...")
def get_gemini_response(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result and result.get("candidates"):
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return "⚠️ No valid response from Gemini."
    except Exception as e:
        return f"⚠️ API error: {str(e)}"

# ------------------------------
# 3. Load Model & Encoder
# ------------------------------
try:
    rf_model = joblib.load("tree_health_model.pkl")
    le = joblib.load("label_encoder.pkl")
    st.success("✅ Model & Encoder loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------------------
# 4. Define Controllable / Non-Controllable
# ------------------------------
non_controllables = [
    "Latitude", "Longitude", "Tree_Height", "DBH",
    "Crown_Width_North_South", "Crown_Width_East_West",
    "Slope", "Elevation", "Gleason_Index", "Menhinick_Index",
    "Temperature", "Humidity"
]
controllables = [
    "Soil_TN", "Soil_TP", "Soil_AP", "Soil_AN",
    "Fire_Risk_Index", "Disturbance_Level"
]

# ------------------------------
# 5. Very Healthy Optimal Ranges
# ------------------------------
optimal_ranges = {
    "Soil_TN": (0.20, 0.25),
    "Soil_TP": (0.08, 0.12),
    "Soil_AP": (0.03, 0.04),
    "Soil_AN": (0.12, 0.15),
    "Fire_Risk_Index": (0.0, 0.05),
    "Disturbance_Level": (0.0, 0.10)
}

# ------------------------------
# 6. Input Fields
# ------------------------------
st.title("🌳 Tree Health Prediction + Chatbot")
fields = {
    "Latitude": (-90.0, 90.0), "Longitude": (-180.0, 180.0),
    "DBH": (1.0, 300.0), "Tree_Height": (10.0, 130.0),
    "Crown_Width_North_South": (0.5, 60.0), "Crown_Width_East_West": (0.5, 60.0),
    "Slope": (0.0, 90.0), "Elevation": (-430.0, 8848.0),
    "Temperature": (-50.0, 60.0), "Humidity": (0.0, 100.0),
    "Soil_TN": (0.01, 2.0), "Soil_TP": (0.01, 1.0),
    "Soil_AP": (0.001, 0.6), "Soil_AN": (0.001, 1.5),
    "Menhinick_Index": (0.0, 10.0), "Gleason_Index": (0.0, 20.0),
    "Disturbance_Level": (0.0, 1.0), "Fire_Risk_Index": (0.0, 1.0)
}

input_data = {}
cols = st.columns(2)
col_idx = 0
for field, (min_val, max_val) in fields.items():
    with cols[col_idx % 2]:
        input_data[field] = st.number_input(
            f"{field} (Min: {min_val}, Max: {max_val})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(min_val),
            step=0.01,
            key=f"input_{field}"
        )
    col_idx += 1

# ------------------------------
# 7. Model Prediction (Reference Only, Not Shown)
# ------------------------------
input_df = pd.DataFrame([input_data])
_ = le.inverse_transform([rf_model.predict(input_df)[0]])[0]  # We don't show this

# ------------------------------
# 8. Final Health Status (Strict Boundaries for Very Healthy)
# ------------------------------
values = {k: input_data[k] for k in controllables}

if all([
    optimal_ranges["Soil_TN"][0] < values["Soil_TN"] < optimal_ranges["Soil_TN"][1],
    optimal_ranges["Soil_TP"][0] < values["Soil_TP"] < optimal_ranges["Soil_TP"][1],
    optimal_ranges["Soil_AP"][0] < values["Soil_AP"] < optimal_ranges["Soil_AP"][1],
    optimal_ranges["Soil_AN"][0] < values["Soil_AN"] < optimal_ranges["Soil_AN"][1],
    values["Fire_Risk_Index"] < optimal_ranges["Fire_Risk_Index"][1],
    values["Disturbance_Level"] < optimal_ranges["Disturbance_Level"][1]
]):
    prediction_label = "Very Healthy"
elif any([
    not (optimal_ranges["Soil_TN"][0] <= values["Soil_TN"] <= optimal_ranges["Soil_TN"][1]),
    not (optimal_ranges["Soil_TP"][0] <= values["Soil_TP"] <= optimal_ranges["Soil_TP"][1]),
    not (optimal_ranges["Soil_AP"][0] <= values["Soil_AP"] <= optimal_ranges["Soil_AP"][1]),
    not (optimal_ranges["Soil_AN"][0] <= values["Soil_AN"] <= optimal_ranges["Soil_AN"][1]),
    values["Fire_Risk_Index"] > optimal_ranges["Fire_Risk_Index"][1],
    values["Disturbance_Level"] > optimal_ranges["Disturbance_Level"][1]
]):
    prediction_label = "Unhealthy"
else:
    prediction_label = "Healthy"

st.success(f"🌳 Final Health Status: **{prediction_label}**")

# ------------------------------
# 9. Debug Table
# ------------------------------
debug_data = []
for param, (low, high) in optimal_ranges.items():
    val = values[param]
    if low < val < high:
        status = "✅ OK"
    elif val == low or val == high:
        status = "⚠️ On Boundary"
    elif val < low:
        status = "⬇️ Low"
    else:
        status = "⬆️ High"
    debug_data.append({"Parameter": param, "Value": val, "Status": status})

debug_df = pd.DataFrame(debug_data)
st.write("### 🔍 Parameter Check")
st.table(debug_df)

# ------------------------------
# 10. Gemini Suggestions
# ------------------------------
if prediction_label != "Very Healthy":
    prompt = f"""
    Current health status: {prediction_label}.
    Controllable parameters and optimal ranges: {optimal_ranges}.
    Current values: {values}.
    Identify values outside the optimal range and suggest exact numeric targets for 'Very Healthy'.
    Format: Parameter | Current Value | Recommended Value
    """
    suggestion = get_gemini_response(prompt)
    st.info(f"🌱 **Adjustments:**\n{suggestion}")
else:
    st.success("✅ Tree is already in Very Healthy condition.")

st.markdown("---")

# ------------------------------
# 11. Open-Ended Gemini Chatbot
# ------------------------------
st.subheader("💬 Tree Health Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)

user_message = st.chat_input("Ask any question about tree health, soil, biodiversity, etc...")
if user_message:
    st.session_state.chat_history.append(("user", user_message))
    chatbot_prompt = f"""
    You are a helpful assistant for a tree health monitoring system.
    Current tree health status: {prediction_label}.
    Current controllable values: {values}.
    The user asks: "{user_message}"

    Provide a detailed, helpful, and expert answer.
    You may include general ecosystem knowledge, soil improvement techniques, 
    biodiversity indexes (Gleason, Menhinick), climate impacts, and actionable advice.
    """
    bot_reply = get_gemini_response(chatbot_prompt)
    st.session_state.chat_history.append(("assistant", bot_reply))
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
