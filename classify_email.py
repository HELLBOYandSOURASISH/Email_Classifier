import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import gensim
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as colors
import gensim.downloader as api
import tensorflow
wv= api.load('word2vec-google-news-300')
def vectorize_review(tokens, model):
    word_vectors = []
    for token in tokens:
        if token in model.key_to_index:  # Check if the token exists in the Word2Vec model's vocabulary
            word_vectors.append(model[token])

    if len(word_vectors) == 0:
        # If no valid word vectors were found, return a zero vector
        return np.zeros(model.vector_size)

    # Average the word vectors to get a single vector for the entire review
    review_vector = np.mean(word_vectors, axis=0)
    return review_vector

# Function to preprocess and predict if an email is spam
def predict_spam(email: str, model):
    # Preprocess the email - You may need to apply the same preprocessing steps as your training data

    pre_processed_email = gensim.utils.simple_preprocess(email)

    # Transform the email using the vectorizer

    email_vec = vectorize_review(pre_processed_email, wv)
    email_vec = np.reshape(email_vec, (1, -1))

    # Predict using the FFNN model
    prediction = model.predict(email_vec)

    # Return whether it's spam (1) or not spam (0)
    if prediction > 0.5:
        return "Spam"
    else:
        return "Not Spam"


import joblib
import keras

# Load models
model_FFNN = keras.models.load_model('spam_email_classifier_FFNN_model.h5')  # For a Keras model
model_rf = joblib.load('spam_email_classifier_random_forest_model.pkl')  # For a Scikit-learn Logistic Regression model
model_xgb = joblib.load('spam_email_classifier_xgboost_model.pkl')  # For a Scikit-learn SVM model


import streamlit as st

# Streamlit UI elements
st.title("Email Spam Prediction with Multiple Models")

# Create a selection box for model choice
model_choice = st.selectbox(
    "Choose a Model for Prediction:",
    ("Random Forest Classification","XGBoost Classification","Feed Forward Neural Network")
)

# Load the corresponding model based on the selection
if model_choice == "Feed Forward Neural Network":
    model = model_FFNN  # Assuming model_FFNN is pre-loaded
elif model_choice == "Random Forest Classification":
    model = model_rf  # Load your Logistic Regression model
elif model_choice == "XGBoost Classification":
    model = model_xgb  # Load your SVM model

# Use st.text_area for multiline input
email_input = st.text_area("Enter the email content:")

if email_input:
    result = predict_spam(email_input, model)
    st.write(f"The email is: {result}")
