import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words]
    # Remove punctuation
    text = ''.join([c for c in text if c not in punctuation])
    return text

def main():
    # Load the SMS dataset
    data = pd.read_csv('sms_spam.csv')
    # Add the message "URGENT: Your account has been locked. Please click the link to unlock it." to the spam category
    data.loc[data['text'] == 'URGENT: Your account has been locked. Please click the link to unlock it.', 'label'] = 'spam'

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)
    # Download the stopwords corpus
    nltk.download('stopwords')
    # Preprocess the text data
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    # Deploy the model
    while True:
        message = input('Enter an SMS message: ')
        message = clean_text(message)
        x = vectorizer.transform([message])
        y_pred = model.predict(x)
        print('Is this message spam?', y_pred[0])

if __name__ == '__main__':
    main()









