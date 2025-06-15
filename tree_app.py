import streamlit as st
import joblib
import openai
import numpy as np
import re

# Load your trained model and label encoder
model = joblib.load("tree_health_model.pkl")
le = joblib.load("label_encoder.pkl")

# Set OpenAI API Key (Use Streamlit secrets or paste directly — not recommended)
openai.api_key = st.secrets["openai_api_key"]

# Title
st.title("🌿 Tree Health AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
user_input = st.chat_input("Ask about your tree health...")

# GPT Function to Extract Features
def extract_features_with_gpt(prompt):
    system_msg = """You are a helpful assistant. 
Given a message about tree properties, extract only numerical features as a Python list of floats.
Example: "Tree is 5m tall, crown is 2.3, DBH is 0.8" → [5.0, 2.3, 0.8]
Only return a list like: [5.0, 2.3, 0.8]
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content": system_msg},
            {"role":"user", "content": prompt}
        ]
    )
    return eval(response.choices[0].message["content"])

# GPT for Answering Questions
def ask_gpt_followup(question, prediction):
    context = f"The tree health prediction is: {prediction}."
    followup = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content": "You are a tree expert AI. Give helpful, short, friendly explanations."},
            {"role":"user", "content": f"{context} {question}"}
        ]
    )
    return followup.choices[0].message["content"]

# Process input
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        features = extract_features_with_gpt(user_input)
        prediction = model.predict([features])
        label = le.inverse_transform(prediction)[0]

        # AI-generated explanation
        gpt_response = ask_gpt_followup(user_input, label)

        full_response = f"**Tree Health:** {label}\n\n{gpt_response}"

    except Exception as e:
        full_response = "Sorry, I couldn't understand that. Please give me more info like tree height, crown diameter, DBH, etc."

    # Show bot reply
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)
