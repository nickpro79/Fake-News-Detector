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

# Load the news dataset
news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')
text = news_dataset.iloc[1000]['text']
# Merge the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Define the stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming to the news content
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Separate the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Convert the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
joblib.dump(model, 'news_model.joblib')
joblib.dump(stemming, 'stemming_function.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

