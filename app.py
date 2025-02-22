import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Download necessary NLTK resources
nltk.download("punkt")

# Load the DistilGPT-2 text-generation model
chatbot = pipeline("text-generation", model="distilgpt2")

# List of common symptoms for basic detection
common_symptoms = [
    "fever", "cough", "headache", "fatigue", "chest pain", "shortness of breath", 
    "dizziness", "nausea", "vomiting", "diarrhea", "sore throat", "muscle pain", 
    "joint pain", "loss of taste", "loss of smell", "rash", "cold", "anxiety"
]

# Function to detect symptoms from user input
def detect_symptoms(user_input):
    words = word_tokenize(user_input.lower())  # Tokenize and normalize text
    detected = [word for word in words if word in common_symptoms]
    return detected

# Chatbot logic
def healthcare_chatbot(user_input):
    user_input = user_input.lower()  # Normalize text

    # Emergency keyword detection
    emergency_keywords = ["chest pain", "difficulty breathing", "severe headache", "heart attack"]
    if any(keyword in user_input for keyword in emergency_keywords):
        return "⚠️ **URGENT:** Please call emergency services or visit the nearest hospital immediately!"

    # Detect symptoms
    symptoms = detect_symptoms(user_input)
    if symptoms:
        return f"I detected possible symptoms: {', '.join(symptoms)}. Please consult a doctor for an accurate diagnosis."

    # Standard responses
    if "symptom" in user_input:
        return "Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit UI
def main():
    # **Ensuring the heading is at the top**
    st.title("🩺 Healthcare Assistant Chatbot")  # **Title at the top**
    st.markdown("Welcome! Ask me about symptoms, medications, or appointments.")

    # **User input field**
    user_input = st.text_input("How can I assist you today?")  

    if st.button("Submit"):
        if user_input:
            st.write("**User:**", user_input)
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:**", response)
        else:
            st.warning("⚠️ Please enter a message to get a response.")

if __name__ == "__main__":
    main()
