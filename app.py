import streamlit as st
import tensorflow as tf
import pickle
from preprocess import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the saved model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sentiment_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


def predict_sentiment(text, model, tokenizer):
    # Clean the text
    cleaned_text = clean_text(text)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    padded = pad_sequences(sequence, maxlen=200, padding="post")

    # Predict
    prediction = model.predict(padded)[0][0]
    return prediction


def main():
    st.title("Movie Review Sentiment Analysis")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model()
    except Exception as e:
        st.error("Error loading model. Please make sure the model is trained first.")
        return

    # Text input
    review_text = st.text_area("Enter your movie review:", height=150)

    if st.button("Analyze Sentiment"):
        if review_text.strip() != "":
            # Get prediction
            prediction = predict_sentiment(review_text, model, tokenizer)

            # Display result
            st.write("### Sentiment Analysis Result:")
            if prediction >= 0.5:
                st.success(f"Positive sentiment (Score: {prediction:.2f})")
            else:
                st.error(f"Negative sentiment (Score: {prediction:.2f})")
        else:
            st.warning("Please enter a review first.")


if __name__ == "__main__":
    main()
