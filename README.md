## Summary of this application

A Python application that is reading the IMDb (Internet Movie Database) dataset and then uses a RNN to do sentiment classification using Tensorflow.
Proper text preprocessing is done. Properly converted into embeddings.
We want to run this application in VS Code on my local Machine.
Save the model into a .h5 file. Use a Streamlit web app as the User Interface and make sure it can be deployed.
Please make sure the RNN is simple and is trained to perform better. Please make sure the application can make a prediction.

## Instructions for running this application:

This will:

Train a simple but effective RNN model on the IMDb dataset
Save the trained model and tokenizer
Provide a web interface where users can input reviews and get sentiment predictions

The model architecture includes:

An embedding layer to convert words to dense vectors
Two LSTM layers for sequence processing
Dense layers with dropout for classification
Binary cross-entropy loss for binary classification

The preprocessing includes:

Cleaning text (removing HTML tags, special characters)
Converting to lowercase
Tokenization
Padding sequences to fixed length

The Streamlit interface provides:

A text input area for new reviews
Sentiment prediction with confidence score
Clear visualization of results

To deploy the application:

Create a GitHub repository and push your code
You can deploy it on Streamlit Cloud, Heroku, or any other platform that supports Python applications
Make sure to include all the files and requirements.txt

The model should achieve reasonable accuracy on the IMDb dataset (typically around 85-90% after a few epochs).
