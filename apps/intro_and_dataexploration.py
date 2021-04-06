import streamlit as st
#st. set_page_config(layout="wide")
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words('english')
porter = nltk.PorterStemmer()

processed_text = pd.read_pickle("pickle-files/dummy.pkl")
vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(processed_text)
load_model = pickle.load(open('pickle-files/svm_model.pkl', 'rb'))

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
#@st.cache(suppress_st_warning=True)
def app(n=5):
    st.title('Spam Filtering Techniques for Short Message Service')

    st.header('Introduction')

    st.subheader('Abstract')

    st.markdown("""Short Message Servive (SMS) is one of the most popular telecommunication service with approximately 15 millions SMS 
    sent each minute around the world in 2017. Spam can be described as unwanted messages sent in bulk to a group of recipients. 
    SMS spamming has become a major nuisance to mobile users because of their intrusive nature and their waste of money, 
    network bandwidth and time. However, SMS spam filtering techniques are still at a relatively early stage due to the limited 
    amount of publicly available datasets. Most of the existing methods inherit from email spam filtering techniques which do not 
    always perform well on SMS spam.""")

    st.markdown("""Here, we study various short message service spam filtering techniques based on a [Kaggle 
    	dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) composed of 5572 messages, whose 4825 are legitimate 
    	and 747 are spam. The [Bag-of-Words](https://en.wikipedia.org/wiki/Bag-of-words_model) models followed by 
    	term-frequency-inverse-document frequency(tf-idf) transformation is employed for feature extraction. 
    	Several state-of-the-art classifiers are compared, i.e. logistic regression, regularized logistic regression, linear 
    	and kernel support vector machine (SVM), k-nearest neighbours, multinomial Bayes and decision trees where the 
    	best hyper-parameters are identified using 10-fold cross validation. We demonstrate that all the classifiers perform 
    	remarkably well in terms of misclassification error and that even simple linear methods, such as logistic regression 
    	leads to less than 5% of misclassification error. We study two resampling methods that can be used to counter the class 
    	imbalance present in the training set, i.e. downsampling of the majority class and upsampling of the minority class. 
    	We show that both lead to an increase of the sensitivity at the cost of a lower specificity.""")

    st.markdown('You can toggle the navigation menu to see the various operations performed on the dataset.')

    st.write('')    	

    st.markdown('Here, you can enter your own text and see what the clasifier tells about it...')
    
    st.markdown("""For example, 
    * `CONGRATlations You won 2 tIckETs to Hamilton in 
	NYC http://www.hamiltonbroadway.com/J?NaIOl/event   wORtH over $500.00...CALL 
	555-477-8914 or send message to: hamilton@freetix.com to get ticket !! !`* would return *`Spam`* and 
    * `Sounds good, Tom, then see u there`* would return *`Not a spam`*""" )

    st.write('')
            
    text = st.text_area('')
    if text != '':
    	if load_model.predict(vectorizer.transform([preprocess_text(text)])):
    		st.write('Spam')
    	else:
    		st.write('Not a spam')

	# stop_words = stopwords.words('english')
	# porter = nltk.PorterStemmer()
	
	# processed_text = pd.read_pickle("./dummy.pkl")
	# vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(processed_text)

	# load_model = pickle.load(open('svm_model.pkl', 'rb'))

	# st.markdown("Let's classify this and see what it tells- WINNER!! This is the secret code to unlock the money: C3421.")
	# text = "WINNER!! This is the secret code to unlock the money- C3421."
	# if load_model.predict(vectorizer.transform([preprocess_text(text)])):
	#     st.write('spam')
	# else:
	#     st.write('not spam')

#app(5)
