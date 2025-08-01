# Run the Streamlit app
streamlit run tree_app.py

# Tree_Model
Predicts the health status of trees (Healthy or Unhealthy) using a machine learning model. Useful for urban forestry, tree retirement planning, and ecological monitoring.


**Overview**
This project predicts the health status of trees and provides specific, actionable recommendations to improve their condition.
The solution uses:
-A Random Forest Classifier to predict tree health
-A Streamlit web interface for user input & display
-Google Gemini API to suggest exact numeric targets for controllable parameters
-An AI-powered chatbot for personalized advice


**Algorithm & Approach**
-Model: Random Forest Classifier (trained offline, loaded via .pkl)
-Prediction Task: Classify tree health as Unhealthy, Healthy, or Very Healthy
-Recommendation Engine: Gemini API analyzes controllable parameters and suggests optimized values to improve tree health
-Chatbot: Handles user queries about improvement, focusing only on controllable factors


**Libraries & Tools**
Python: Main programming language
Libraries:
pandas – Data manipulation
numpy – Numerical computations
scikit-learn – Model training & prediction
joblib – Model and encoder loading
streamlit – Web app interface
requests, json – API integration with Gemini


**How It Works**
-User Inputs parameters in Streamlit (19 environmental and biological values)
-Model Predicts the health category (Unhealthy, Healthy, Very Healthy)
-If health is below Very Healthy:
-Gemini API Suggests numeric adjustments for controllable parameters
-Chatbot answers follow-up questions on improving tree health


** Input & Output**
--Input:
19 parameters (both controllable & non-controllable)

--Output Categories:
⚠️ Unhealthy – Tree at risk, needs attention
🌱 Healthy – Tree is in good condition
✅ Very Healthy – Tree is in optimal condition


**The app considers multiple environmental and biological inputs, including:**
Controllable Factors: Temperature, Humidity, Soil nutrients (TN, TP, AP, AN), Fire Risk, Disturbance Level
Fixed Factors: Tree Height, DBH, Location, Slope, Elevation (used for prediction but not for direct adjustments)
