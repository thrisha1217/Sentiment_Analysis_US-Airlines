# app.py

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download NLTK data (if not already present) ---
# This is important for the text preprocessing function
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK data (stopwords, wordnet)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    st.success("NLTK data downloaded.")

# --- Text Preprocessing Function ---
# This MUST be the same function you used to train your model
def preprocess_text(text):
    """Cleans and preprocesses text for the model."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|@\S+", "", text)  # Remove URLs and mentions
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()  # Keep only letters, lowercase
    tokens = text.split()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# --- Load Your Trained Model and Vectorizer ---
# Use a try-except block to handle potential FileNotFoundError
try:
    with open('models/enhanced_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('models/enhanced_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model files not found! Please make sure 'enhanced_model.pkl' and 'enhanced_vectorizer.pkl' are in the 'models' directory.")
    # Stop the app if models can't be loaded
    st.stop()

# --- Streamlit App Layout ---

# Set the title and a descriptive subtitle
st.title("✈️ US Airline Sentiment Analyzer")
st.markdown("This app uses a custom-trained Machine Learning model to predict the sentiment of airline-related text. It was trained on a combined dataset of over 140,000 tweets and customer reviews.")
st.markdown("---")

# Create a text area for user input
user_input = st.text_area(
    "Enter a tweet or a review to analyze:",
    "The flight was fantastic and the crew was very professional!",
    height=150
)

# Create a button to trigger the analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Preprocess the user's input
        clean_input = preprocess_text(user_input)
        
        # 2. Vectorize the cleaned input
        vectorized_input = vectorizer.transform([clean_input])
        
        # 3. Predict using your loaded model
        prediction = model.predict(vectorized_input)[0]
        
        # Display the result
        st.subheader("Analysis Result")
        if prediction == "positive":
            st.success("The sentiment is **Positive** 👍")
        else:
            st.error("The sentiment is **Negative** 👎")
            
    else:
        st.warning("Please enter some text to analyze.")

# Add a footer or some info about the model
st.markdown("---")
st.info("Model: Support Vector Machine (SVM) with TF-IDF Vectorizer.")