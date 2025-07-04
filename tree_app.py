import streamlit as st # Imports the Streamlit library for building web applications
import pandas as pd # Imports Pandas for data manipulation (e.g., creating DataFrames)
import joblib # Imports joblib for loading pre-trained machine learning models and encoders
import requests # Imports the requests library for making HTTP requests to external APIs (like Gemini)
import json # Imports json for handling JSON data, which is used for API communication

# --- Configuration ---
# This section defines global configuration variables for the application.
# IMPORTANT: For a production deployment, replace this with a secure method
# of handling your API key, e.g., Streamlit Secrets or environment variables.
# For local testing, you can put your actual API key here, but DO NOT commit it to public repos.
GEMINI_API_KEY = "AIzaSyBWYikFZtblzgel33kCbzE0Ibl1qjt887o" # Variable to store the Gemini API key. It's left empty here as Canvas provides it at runtime.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" # The endpoint URL for the Gemini API.

# 🔹 Get response from Gemini model
# This function interacts with the Gemini API to get a conversational response.
@st.cache_data(show_spinner="Thinking...") # Caches the function's output to improve performance and shows a spinner while running.
def get_gemini_response(prompt):
    """
    Sends a prompt to the Gemini API and returns the generated text.
    Uses requests library for HTTP communication.
    """
    # Check if the Gemini API key is configured.
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is not configured. Please set GEMINI_API_KEY.")
        return "⚠️ Gemini API key missing."

    # Define headers for the HTTP request, specifying content type as JSON.
    headers = {
        'Content-Type': 'application/json',
    }
    # Define parameters for the HTTP request, including the API key.
    params = {
        'key': GEMINI_API_KEY,
    }
    # Define the payload (body) of the HTTP request, structuring the prompt for the Gemini API.
    payload = {
        "contents": [
            {
                "role": "user", # Specifies the role of the sender (user in this case)
                "parts": [
                    {"text": prompt} # The actual text prompt to send to the AI model
                ]
            }
        ]
    }

    try:
        # Make the POST request to the Gemini API with the defined URL, headers, parameters, and payload.
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx) to catch network/API errors.
        result = response.json() # Parse the JSON response from the API.

        # Check if the response contains valid generated text.
        if result and result.get('candidates'):
            # Extract the text from the first candidate's first part of the response.
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            return generated_text.strip() # Return the cleaned (whitespace removed) generated text.
        else:
            # Warn if the API response structure is unexpected.
            st.warning(f"⚠️ Gemini API response did not contain expected content: {result}")
            return "⚠️ Could not get a valid response from Gemini API."
    except requests.exceptions.RequestException as e:
        # Handle errors specifically related to HTTP requests (e.g., network issues, invalid URL).
        st.error(f"Error communicating with Gemini API: {e}")
        return f"⚠️ API communication error: {str(e)}"
    except Exception as e:
        # Handle any other unexpected errors during response processing.
        st.error(f"An unexpected error occurred while processing Gemini response: {e}")
        return f"⚠️ Unexpected error: {str(e)}"

# 🔹 Load Model and Label Encoder
# This block attempts to load the pre-trained machine learning model
# and the label encoder, which are essential for making predictions.
try:
    # Load the Random Forest model from the 'tree_health_model.pkl' file.
    # Ensure this path is correct relative to where your Streamlit app runs.
    rf_model = joblib.load("tree_health_model.pkl")
    # Load the LabelEncoder from the 'label_encoder.pkl' file.
    # This encoder is used to convert numerical predictions back to human-readable labels.
    le = joblib.load("label_encoder.pkl")
    st.success("✅ ML Model and Label Encoder loaded successfully.") # Display a success message if models load correctly.
except Exception as e:
    # If loading fails, display an error message and stop the Streamlit app.
    st.error(f"Error loading model or label encoder. Please ensure 'tree_health_model.pkl' and 'label_encoder.pkl' are in the same directory as the script: {e}")
    st.stop() # Stop the app if models can't be loaded, as it cannot function without them.

# 🔐 User Credentials
# This dictionary stores hardcoded usernames and passwords for a simple login system.
# In a real-world application, this would be replaced with a secure authentication method
# (e.g., database integration, OAuth).
users = {
    "admin": "password123",
    "akash": "treehealth2025"
}

# 🔐 Initialize Session State
# Streamlit's session_state is used to persist variables across reruns of the app.
# This is crucial for maintaining application state (like login status and chat history)
# as Streamlit reruns the script from top to bottom on every user interaction.
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False # Initializes the authentication status to False if not already set.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Initializes an empty list to store past chat interactions.
# Initialize chat_input in session state if it doesn't exist.
# This is used to control the value of the chat input text box, allowing it to be cleared.
if 'chat_input_value' not in st.session_state:
    st.session_state.chat_input_value = "" # Sets the initial value of the chat input to an empty string.

# 🔐 Login Page
# This function defines the layout and logic for the user login interface.
def login():
    st.title("🔒 Login to Tree Health App") # Displays the title for the login page.
    st.markdown("---") # Adds a horizontal rule for visual separation.
    username = st.text_input("Username (max 20 chars)", max_chars=20) # Text input for username with a character limit.
    password = st.text_input("Password (max 15 chars)", type="password", max_chars=15) # Password input, masked for security, with a character limit.

    if st.button("Login", help="Click to log in"): # Login button with a tooltip.
        # Input validation for username and password length.
        if len(username) > 20:
            st.warning("❗ Username must be at most 20 characters.")
        elif len(password) > 15:
            st.warning("❗ Password must be at most 15 characters.")
        # Check if provided credentials match the stored users.
        elif username in users and users[username] == password:
            st.session_state.authenticated = True # Set authentication status to True upon successful login.
            st.success("✅ Login successful! Redirecting to prediction page...") # Display success message.
            st.rerun() # Rerun the app to switch to the prediction page, as the `authenticated` state has changed.
        else:
            st.error("❌ Invalid username or password.") # Display error for incorrect credentials.

# Callback function to clear the input
# This function is specifically designed to be called by Streamlit widgets' `on_click` event.
# It modifies the `chat_input_value` in session state, which will then clear the text input
# widget in the subsequent rerun.
def clear_chat_input():
    st.session_state.chat_input_value = ""

# 🌳 Prediction Page
# This function defines the main application interface for tree health prediction and chatbot.
# It is displayed only after successful user authentication.
def prediction_page():
    st.title("🌳 Tree Health Prediction App") # Title for the prediction page.
    st.markdown("---") # Horizontal rule.
    st.header("📊 Enter Tree Details for Prediction") # Section header for data input.

    # Define the input fields and their valid ranges for tree features.
    # This dictionary maps feature names to their acceptable minimum and maximum values.
    fields = {
        "Latitude": (-90.0, 90.0), "Longitude": (-180.0, 180.0),
        "DBH": (1.0, 300.0), "Tree_Height": (1.0, 130.0),
        "Crown_Width_North_South": (0.5, 60.0), "Crown_Width_East_West": (0.5, 60.0),
        "Slope": (0.0, 90.0), "Elevation": (-430.0, 8848.0),
        "Temperature": (-50.0, 60.0), "Humidity": (0.0, 100.0),
        "Soil_TN": (0.01, 2.0), "Soil_TP": (0.01, 1.0),
        "Soil_AP": (0.001, 0.6), "Soil_AN": (0.001, 1.5),
        "Menhinick_Index": (0.0, 10.0), "Gleason_Index": (0.0, 20.0),
        "Disturbance_Level": (0.0, 1.0), "Fire_Risk_Index": (0.0, 1.0)
    }

    input_data = {} # Dictionary to store the numerical inputs from the user.
    # Use Streamlit columns for better layout of input fields, arranging them in two columns.
    cols = st.columns(2) # Create two columns for inputs.
    col_idx = 0
    for field, (min_val, max_val) in fields.items():
        with cols[col_idx % 2]: # Place input in alternating columns (0, 1, 0, 1...).
            input_data[field] = st.number_input(
                f"{field} (Min: {min_val}, Max: {max_val})", # Label for the number input widget.
                min_value=float(min_val), # Minimum allowed value for the input.
                max_value=float(max_val), # Maximum allowed value for the input.
                value=float(min_val), # Set a default value within the specified range.
                step=0.01, # Allow decimal inputs with a step of 0.01.
                key=f"input_{field}" # Unique key for each input widget, essential for Streamlit.
            )
        col_idx += 1

    st.markdown("---") # Separator after the input fields.

    # Button to trigger the ML model prediction.
    if st.button("🚀 Predict Tree Health", help="Click to get the health prediction based on inputs"):
        try:
            # Create a Pandas DataFrame from the collected user inputs.
            # It's crucial that the column order here matches the order of features
            # used when the ML model was trained to ensure correct prediction.
            input_df = pd.DataFrame([input_data])

            # Make prediction using the loaded Random Forest model.
            prediction_encoded = rf_model.predict(input_df)[0]
            # Decode the numeric prediction back to a human-readable label (e.g., "Healthy", "Unhealthy").
            prediction_label = le.inverse_transform([prediction_encoded])[0]

            st.success(f"🌳 Predicted Health Status: **{prediction_label}**") # Display the predicted health status.

            # Define a dictionary of treatment recommendations based on the predicted health status.
            treatment_recommendations = {
                "Unhealthy": "The tree appears unhealthy. Consider checking for pests, diseases, and nutrient deficiencies. Ensure proper watering and drainage. It's highly recommended to consult a certified arborist for a professional diagnosis and tailored treatment plan.",
                "Healthy": "The tree appears healthy! Continue with regular maintenance, including appropriate watering during dry periods, mulching, and occasional inspection for early signs of stress. Good job!",
                "Very Healthy": "Excellent! The tree is very healthy. Maintain current care practices. Focus on preventative measures like proper pruning, ensuring good air circulation, and protecting the tree from environmental stressors. Keep up the great work!"
            }
            # Display the relevant treatment recommendation.
            st.info(f"💡 **Treatment Recommendation:** {treatment_recommendations.get(prediction_label, 'No specific recommendation available for this status.')}")

        except Exception as e:
            # Display an error if the prediction process fails (e.g., due to incorrect input format).
            st.error(f"Prediction failed: {e}. Please check your inputs and try again.")

    st.markdown("---") # Separator before the chatbot section.
    st.subheader("💬 Ask Anything about Tree Health (Powered by Gemini AI)") # Header for the chatbot.
    # Text input for the user to ask questions to the Gemini chatbot.
    # `key="chat_input"` uniquely identifies this widget, allowing Streamlit to manage its state.
    # `value=st.session_state.chat_input_value` links the input's current value
    # to a session state variable, which is used by the `clear_chat_input` callback.
    user_question = st.text_input(
        "Your question about tree health:",
        key="chat_input", # This key identifies the widget
        value=st.session_state.chat_input_value # This sets its initial value from session state
    )

    # Button to send the user's question to the Gemini AI.
    # `on_click=clear_chat_input` ensures the input box is cleared
    # when the button is pressed, before the next Streamlit rerun, preventing the API error.
    if st.button(
        "Ask AI",
        help="Get an AI-powered answer to your tree health question",
        on_click=clear_chat_input # Call this function when the button is clicked
    ) and user_question: # Only proceed if there's a question in the input field.
        # Construct a more specific prompt for Gemini to guide its response,
        # instructing it to act as a "helpful tree health expert".
        full_prompt = f"You are a helpful tree health expert. Provide a concise and informative answer to the following question:\n\nQuestion: {user_question}"
        # Get the response from the Gemini API using the `get_gemini_response` function.
        response = get_gemini_response(full_prompt)
        # Add the user's question and Gemini's response to the chat history in session state.
        st.session_state.chat_history.append((user_question, response))
        # The input will be cleared by the on_click callback in the next rerun, so no direct clearing here.

    st.subheader("Chat History") # Header for displaying chat history.
    # Display the chat history.
    if not st.session_state.chat_history:
        st.info("No chat history yet. Ask a question above!") # Message if chat history is empty.
    else:
        # Display the most recent 5 interactions from the chat history, in reverse chronological order.
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
            st.markdown(f"**🧑‍💻 You:** {q}") # Display user's question.
            st.markdown(f"**🤖 Gemini AI:** {a}") # Display AI's answer.
            st.markdown("---") # Separator for individual chat turns.

# 🔁 Run App based on authentication status
# This is the main control flow of the Streamlit application.
# It checks the `authenticated` status in `st.session_state` and displays
# either the login page or the main prediction page accordingly.
if not st.session_state.authenticated:
    login() # Show the login page if the user is not authenticated.
else:
    prediction_page() # Show the prediction page if the user is authenticated.
