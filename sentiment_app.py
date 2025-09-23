import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to analyze sentiment and return a friendly result
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the text using VADER and returns a classification
    and the compound score.
    """
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    compound_score = score['compound']

    if compound_score >= 0.05:
        return "Positive", compound_score, "😊"
    elif compound_score <= -0.05:
        return "Negative", compound_score, "😞"
    else:
        return "Neutral", compound_score, "😐"

# --- Streamlit App Layout ---

# Set the title of the app
st.title("Sentiment Analysis Web App 💬")
st.markdown("Enter any text below to determine its sentiment. This app uses the VADER model, which is specifically tuned for social media and informal text.")

# Create a text area for user input
user_text = st.text_area("Enter your text here:", "I love using Streamlit! It's so easy and fun.", height=150)

# Create a button to trigger the analysis
if st.button("Analyze Sentiment"):
    if user_text:
        # Get the sentiment analysis results
        sentiment, compound_score, emoji = analyze_sentiment(user_text)

        # Display the result in a styled box
        st.subheader("Analysis Result")
        if sentiment == "Positive":
            st.success(f"The sentiment is **{sentiment}** {emoji}")
        elif sentiment == "Negative":
            st.error(f"The sentiment is **{sentiment}** {emoji}")
        else:
            st.info(f"The sentiment is **{sentiment}** {emoji}")

        # Show a more detailed score
        st.write(f"**Compound Score:** `{compound_score}` (A score from -1 to +1)")
        st.markdown("---")
        st.write("The **compound score** is a single metric that summarizes the text's sentiment. Scores > 0.05 are positive, scores < -0.05 are negative, and the rest are neutral.")

    else:
        st.warning("Please enter some text to analyze.")