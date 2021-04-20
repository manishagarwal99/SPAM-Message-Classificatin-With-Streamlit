import streamlit as st
import re
import nltk
from nltk.corpus import stopwords

#st. set_page_config(layout="wide")

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

#@st.cache(suppress_st_warning=True)
def app(n=5):
	st.header('Text Pre-processing')

	st.markdown("### Normalization")
	st.markdown(""" Let's take an example text and move forward so it would be easier to understand.""")
	st.write("""**`CONGRATlations You won 2 tIckETs to Hamilton in 
	NYC http://www.hamiltonbroadway.com/J?NaIOl/event   wORtH over $500.00...CALL 
	555-477-8914 or send message to: hamilton@freetix.com to get ticket !! !`**""")
	st.markdown("""I would definitely deem this as spam. But clearly there's a lot going on here: phone 
	numbers, emails, website URLs, money amounts, and gratuitous whitespace and punctuation. Some terms 
	are randomly capitalized, others are in all-caps. Since these terms might show up in any one of the 
	training examples in countless forms, we need a way to ensure each training example is on equal footing 
	via a preprocessing step called **normalization**.""")
	st.markdown("""
	Instead of removing the following terms, for each training example, 
	let's replace them with a specific string:
	* Replace `email addresses` with `'emailaddr'`
	* Replace `URLs` with `'httpaddr'`
	* Replace `money symbols` with `'moneysymb'`
	* Replace `phone numbers` with `'phnnum'`
	* Replace `numbers` with `'num'`""")
	st.markdown("""Next, we'll remove all punctuation since `today` and `today?` refer to the same word. In addition, 
	let's `collapse all whitespace` (spaces, line breaks, tabs) into a single space. 
	Furthermore, we'll `eliminate any leading or trailing whitespace`.
	We'll also need to treat the words "there", "There" and "ThERe" as the same word. 
	Therefore, let's `lowercase the entire corpus`.""")
	st.markdown('The example text has now become :')
	st.write("""**`congratlations you won numbr tickets to hamilton in nyc httpaddr 
	worth over moneysymbnumbr call phonenumbr or send message to emailaddr to get ticket`**""")
	
	st.markdown("### Removing stop words")
	st.markdown("""Some words in the English language, while necessary, don't contribute much to the 
	meaning of a phrase. These words, such as `when`, `had`, `those` or `before`, are called 
	**stop words** and should be filtered out. The Natural Language Toolkit (NLTK), a popular 
	Python library for NLP, provides common stop words.
	This list of stop words is literally stored in a Python *list*. 
	If instead we convert it to a Python *set*, iterating over the stop words will go *much* faster, 
	and saves time off this preprocessing step.""")
	st.markdown('The example text has now become :')
	st.write("""**`congratlations numbr tickets hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send message emailaddr get ticket`**""")

	st.markdown("### Stemming & Lemmatization")
	st.markdown(""" We use *Stemming* and *Lemmatization* to reduce the number of tokens that carry out the same information and hence speed up the whole process. 
	It's likely the corpus contains words with various suffixes such as "distribute", 
	"distributing", "distributor" or "distribution". We can replace these four words with just 
	"distribut" via a preprocessing step called **stemming**.
	There are numerous stemming strategies. Let's use one from NLTK called the `Porter stemmer`.""")
	st.write("""The example text after *stemming* has now become : """)
	st.markdown("`congratl numbr ticket hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send messag emailaddr get ticket`")
	st.markdown("""
	However, stemming can be crude and chop off suffixes haphazhardly. 
	The difference is that stem might not be an actual word whereas, lemma is an actual language word. 
	A better alternative is **lemmatization**. Here, language is important so we use lemmatization as it uses a corpus to match root forms. 
	For example, the word `worse` reduces to `bad`.""")
	st.markdown('The example text after *lemmatization* has now become :')
	st.write("""`congratulations numbr ticket hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send message emailaddr get ticket`""")
	st.markdown("Notice the change in *congratulations* and *message*.")
	st.write('')
	
	st.markdown("""
	Now that we've performed all operations on every data and enriched the corpus 
	with meaningful terms, we're ready to move on to *feature engineering*.""")

	st.markdown("""Here you can enter your sentence and see the processed output :""")
	text = st.text_area('')
	if text != '':
		st.write(preprocess_text(text))	

#app(5)
