import streamlit as st
from PIL import Image

#st. set_page_config(layout="wide")

def app(n=5):
	st.header('Data Exploration')
	st.write('')

	row1, row2 = st.beta_columns((2,1))

	row1.markdown("### General information on the dataset")
	row1.markdown("""The study is based on the “SMS Spam Collection Dataset” available on [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset). The dataset is composed of 5572 SMS 
	tagged according to being ham (legitimate) or spam. It contains
	4825 ham messages and 747 spam messages representing 86%
	and 14% of the dataset respectively. Thus, the problem is
	quite imbalanced which may have impact on the sensitivity
	and specificity of the classifiers.""")
	image1 = Image.open('data-explore-pics/num_ham_vs_spam.png')
	row2.image(image1, caption='Fig. 1. Ham vs Spam data')

	row1, row2 = st.beta_columns((1,2))

	row2.markdown("### Most frequent words in the dataset")
	row2.markdown("""In a first data exploratory step, we display a bar plot of
	the 30 most frequent words in the dataset in Figure 2. It can
	be noticed they are all commonly used english words, usually
	called stop words.""")
	image2 = Image.open('data-explore-pics/num_occur.png')
	row1.image(image2, caption='Fig. 2. 30 most frequent words')

	st.markdown("### Wordclouds")
	st.markdown("""The main idea behind SMS filtering relies on the fact
	that spam and ham messages are composed of different
	words (or groups of words). More precisely, some words are
	very likely to occur more frequently in ham messages than
	in spam messages and vice-versa. To illustrate such a fact,
	Figs. 3(a) and 3(b) display word clouds for ham and spam
	messages respectively, where the most frequent words appear
	the biggest.
	Interestingly, it can be seen that most frequent words
	in spam messages are related to money (“free”, “account”,
	“winner”, “win”, credit, "service") and urgency (“urgent”,
	“last”, “week”), which must be familiar to anybody
	that have already received spam messages. In ham messages,
	the words are the usual ones used in discussion like “anything”,
	“home”, “go”, “wait”, “remember” and “joke”. We also
	notice the presence of slang words that are
	characteristics of text messages like “rofl”, “dat”, "bitching",
	“lol” and “yup”.""")
	row1, row2 = st.beta_columns(2)

	image3 = Image.open('data-explore-pics/ham_words.png')
	row1.image(image3, caption='(a) Ham Wordcloud')

	image4 = Image.open('data-explore-pics/spam_words.png')
	row2.image(image4, caption='(b) Spam Wordcloud')
	row = st.beta_columns((1,2.5,1))
	row[1].markdown('Fig. 3. Wordcloud of words in (a)-Ham messages and (b)-Spam messages.')
	st.write('')
	st.write('')

	st.markdown("### Inference")
	st.markdown("""
	The above mentioned plots are informative in terms of two
	necessary preprocessing steps:
	* Normalization
	* Remove the most frequent english words 
	* Stemming and Lematization of words""")

#app(5)
