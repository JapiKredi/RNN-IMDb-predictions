import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np
from preprocess import clean_text, prepare_data
from model import create_model

def train_model():
    # Load IMDb dataset
    print("Loading IMDb dataset...")
    max_words = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    
    # Convert indices back to words
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Decode reviews
    decoded_reviews_train = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) 
                           for sequence in x_train]
    decoded_reviews_test = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) 
                          for sequence in x_test]
    
    # Clean and preprocess text
    print("Preprocessing data...")
    cleaned_train = [clean_text(text) for text in decoded_reviews_train]
    cleaned_test = [clean_text(text) for text in decoded_reviews_test]
    
    # Prepare sequences
    x_train_padded = prepare_data(cleaned_train)
    x_test_padded = prepare_data(cleaned_test)
    
    # Create and train model
    print("Creating and training model...")
    model = create_model(max_words)
    
    history = model.fit(x_train_padded, y_train,
                       epochs=5,