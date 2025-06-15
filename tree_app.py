import streamlit as st
from transformers import pipeline
import joblib

# Load AI model from Hugging Face (FLAN-T5 - free)
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

# Load your trained ML model and label encoder
model = joblib.load("tree_health_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Extract features
def extract_tree_features(user_input):
    prompt = f"Extract tree height, crown width, and DBH in meters from this sentence: {user_input}. Return three numbers separated by commas."
    result = qa_model(prompt, max_new_tokens=50)[0]['generated_text']
    try:
        numbers = [float(x.strip()) for x in result.split(",")]
        if len(numbers) == 3:
            return numbers
    except:
        return None
    return None

# Predict tree health
def predict_health(features):
    prediction = model.predict([features])
    return encoder.inverse_transform(prediction)[0]

# Answer general questions
def answer_tree_question(user_input):
    prompt = f"Answer this tree-related question briefly: {user_input}"
    return qa_model(prompt, max_new_tokens=100)[0]['generated_text']

# Handle complaints
def handle_complaint(user_input):
    keywords = ["not working", "error", "issue", "problem", "slow", "upload", "bug"]
    if any(k in user_input.lower() for k in keywords):
        return "⚠️ Thank you for your feedback! Our technical team has been notified. You can also reach us at support@treecare.ai"
    return None

# Streamlit UI
st.set_page_config(page_title="🌳 Tree Health AI Chatbot")
st.title("🌿 Smart Tree Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask a question or describe your tree (height, DBH, crown width)...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    response = ""

    # Try ML prediction
    features = extract_tree_features(user_input)
    if features:
        prediction = predict_health(features)
        response = f"✅ Based on your tree data, the predicted health is: **{prediction}**"
    else:
        # If not tree input, check complaint
        complaint = handle_complaint(user_input)
        if complaint:
            response = complaint
        else:
            response = answer_tree_question(user_input)

    st.chat_message("assistant").markdown(response)
    st.session_state.history.append({"role": "assistant", "content": response})

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
