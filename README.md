**Project Overview:**

This project predicts the health status of trees and provides data-driven recommendations to improve their condition.

**The system uses:**:

-Random Forest Classifier (trained offline, optimized to avoid overfitting)
-Streamlit Web App for interactive user input
-Google Gemini API to generate numeric recommendations for improvement
-AI Chatbot to answer follow-up queries about health improvement

**Objective**
Predict tree health accurately (Unhealthy, Healthy, Very Healthy)
Identify controllable factors affecting tree health
Provide targeted improvement recommendations

**Algorithm:**
Random Forest Classifier

**Reason for selection:**
Works well with tabular environmental and biological data
Handles non-linear relationships
Robust to outliers and noise

**Predicts health status as:**
Unhealthy – Tree at risk
Healthy – Tree in good condition
Very Healthy – Tree in optimal condition

**Recommendation Engine**
Gemini API analyzes controllable parameters
Suggests numeric target ranges to reach Very Healthy status
Chatbot
AI-powered responses
Answers improvement questions using only controllable factors

**Libraries & Tools**
**Programming Language:**
Python

**Libraries:**
pandas – Data manipulation
numpy – Numerical computations
scikit-learn – Model training and evaluation
joblib – Save/load model and encoder
streamlit – Web interface
requests, json – API communication with Gemini

**Parameters Considered**

**Controllable Factors**
(Can be adjusted through care and management)
Soil_TN (Total Nitrogen)
Soil_TP (Total Phosphorus)
Soil_AP (Available Phosphorus)
Soil_AN (Available Nitrogen)
Fire Risk Index
Disturbance Level

**Non-Controllable Factors**
Latitude, Longitude
Tree Height
DBH (Diameter at Breast Height)
Crown Width (NS/EW)
Slope
Elevation
Temperature (macro-climate)
Humidity (macro-climate)
Gleason Index
Menhinick Index

**How It Works
Workflow**
1)User inputs 19 parameters in Streamlit UI
2)Model predicts health category using Random Forest
3)Rule-based adjustment:
-If parameters are far outside “Healthy” ranges → Override to Unhealthy
-If all parameters fall in “Very Healthy” ranges → Override to Very Healthy
4)Gemini API provides exact numeric adjustments for controllable parameters
5)Chatbot responds to improvement questions

**Input & Output
Input:**
-19 parameters (environmental and biological)

**Output:**
Unhealthy – Needs action
Healthy – Acceptable condition
Very Healthy – Optimal condition

**How to Run the App**
# Clone repository
git clone https://github.com/username/tree-health-prediction.git
cd tree-health-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run tree_app.py
