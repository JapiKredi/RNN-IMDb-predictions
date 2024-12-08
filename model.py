import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def create_model(vocab_size, embedding_dim=100, max_len=200):
    """Create and return the RNN model"""
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model
