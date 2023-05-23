import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from fake_news_classifier import stemming
from keras.models import load_model
from DANN import preprocess_text
dann_model = load_model('dann_model.h5')

# Load the trained model from the file
model = joblib.load('news_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')


# Define a function for predicting whether a news is real or fake
def predict_news(news_text):
    # Apply stemming to the news content
    stemmed_text = stemming(news_text)

    # Convert the news text to numerical data
    X_new = vectorizer.transform([stemmed_text])

    # Make a prediction using the trained logistic regression model
    prediction = model.predict(X_new)

    # Return the prediction
    if prediction[0] == 0:
        return 'Real'
    else:
        return 'Fake'

# Define a function for predicting whether a news is real or fake, and rating the prediction
def predict_news_rating(news_text):
    # Apply stemming to the news content
    stemmed_text = stemming(news_text)

    # Convert the news text to numerical data
    X_new = vectorizer.transform([stemmed_text])

    # Make a prediction using the trained logistic regression model
    prediction = model.predict(X_new)
    proba = model.predict_proba(X_new)[0]

    # Calculate the rating based on the confidence of the model's prediction
    rating = int(round(proba.max() * 5))

    # Return the prediction and rating
    if prediction[0] == 0:
        return 'Real', rating
    else:
        return 'Fake', rating


        
news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')
text = news_dataset.iloc[1000]['text']
print(predict_news_rating(text))
print(news_dataset.iloc[1000]['label'])

def predict_label(text):
    # Preprocess the text
    text_sequence = preprocess_text(text)

    # Predict the label using the DANN model
    predicted_label = dann_model.predict(text_sequence)
    predicted_label = np.argmax(predicted_label, axis=-1)
    
    # Convert predicted label to 0 or 1
    if predicted_label == 1:
        predicted_label = "true"
    else:
        predicted_label = "fake"
    
    # Ask user for rating
   # rating = input("Please rate the prediction from 1 to 5 (1 being low confidence, 5 being high confidence): ")
    #rating = int(rating)
    #prediction = model.predict(X_new)
    #proba = model.predict_proba(X_new)[0]
    
    return (predicted_label)


