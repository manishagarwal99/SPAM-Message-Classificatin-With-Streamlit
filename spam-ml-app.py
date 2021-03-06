import streamlit as st
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
from collections import Counter
# import os
# from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from sklearn.model_selection import(train_test_split,learning_curve,StratifiedShuffleSplit, GridSearchCV,cross_val_score)

st.write("""
# Building a Spam Filter with Naive Bayes!
""")

stop_words = stopwords.words('english')
porter = nltk.PorterStemmer()
def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     cleaned)
    cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
    cleaned = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
    return ' '.join(
        porter.stem(term) 
        for term in cleaned.split()
        if term not in set(stop_words)
    )
processed2 = pd.read_pickle("./dummy.pkl")
vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(processed2)

load_model = pickle.load(open('svm_model.pkl', 'rb'))

st.subheader("Let's classify this and see what it tells : 'WINNER!! This is the secret code to unlock the money: C3421.'")
text = "WINNER!! This is the secret code to unlock the money: C3421."
if load_model.predict(vectorizer.transform([preprocess_text(text)])):
    st.write('spam')
else:
    st.write('not spam')


text = "Sounds good, Tom, then see u there"
if load_model.predict(vectorizer.transform([preprocess_text(text)])):
    st.write('spam')
else:
    st.write('not spam')
st.subheader("Let's classify this and see what it tells")
text = st.text_area("Enter your own text :")
if load_model.predict(vectorizer.transform([preprocess_text(text)])):
    st.write('spam')
else:
    st.write('not spam')