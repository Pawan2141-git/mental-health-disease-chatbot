import streamlit as st
from transformers import pipeline
import pandas as pd
import datetime
import altair as alt
from PIL import Image
import torch
from torchvision import transforms

# Load multi-emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Load image classification pipeline (example: use a general image classifier)
# For demonstration, we use a general image classifier like 'google/vit-base-patch16-224'
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Emotion-based suggestions with emojis
emotion_suggestions = {
    "joy": "😊 That's wonderful! Keep doing what makes you happy.",
    "sadness": "😔 I'm sorry you're feeling sad. Consider talking to a friend or trying some relaxation techniques.",
    "anger": "😠 Feeling angry is natural. Try some deep breathing or a short walk to calm down.",
    "fear": "😨 It's okay to feel scared sometimes. Mindfulness exercises might help you feel grounded.",
    "surprise": "😲 Surprises can be exciting or unsettling. Take a moment to process your feelings.",
    "disgust": "🤢 If something is bothering you, try to focus on positive activities or talk it out.",
    "neutral": "😐 Thanks for sharing. Remember to take care of yourself every day."
}

# Initialize session state for mood tracking
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

def get_top_emotion(emotion_scores):
    sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
    return sorted_emotions[0]['label']

def main():
    st.set_page_config(page_title="Mental Health & Image Analysis Chatbot", page_icon="🧠", layout="centered")
    st.title("🧠 Mental Health & Image Analysis Chatbot")
    st.write("Share how you're feeling or upload an image for analysis.")

    # Text input for emotion analysis
    user_input = st.text_area("How are you feeling?", height=100)

    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing your emotions..."):
                results = emotion_classifier(user_input)[0]
                top_emotion = get_top_emotion(results)
                st.markdown(f"### Detected Emotion: {top_emotion.capitalize()}")
                suggestion = emotion_suggestions.get(top_emotion, "Thank you for sharing.")
                st.info(suggestion)

                # Log mood with timestamp
                st.session_state.mood_log.append({"date": datetime.date.today(), "emotion": top_emotion})

    # Image upload for disease detection
    st.markdown("---")
    st.subheader("🖼️ Upload an Image for Disease Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                preds = image_classifier(image)
                # Show top 3 predictions
                st.markdown("### Image Classification Results:")
                for pred in preds[:3]:
                    st.write(f"{pred['label']}: {pred['score']:.2f}")

    # Show mood tracking chart if we have data
    if st.session_state.mood_log:
        st.subheader("📊 Your Mood Over Time")
        df = pd.DataFrame(st.session_state.mood_log)
        mood_counts = df.groupby(["date", "emotion"]).size().reset_index(name="count")

        chart = alt.Chart(mood_counts).mark_bar().encode(
            x="date:T",
            y="count:Q",
            color="emotion:N",
            tooltip=["date:T", "emotion:N", "count:Q"]
        ).properties(width=700, height=300)

        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.markdown("**Remember:** You're not alone. Seeking help is a sign of strength. 💙")

if __name__ == "__main__":
    main()