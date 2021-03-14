import streamlit as st
#st. set_page_config(layout="wide")

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
	st.markdown("""It's likely the corpus contains words with various suffixes such as "distribute", 
	"distributing", "distributor" or "distribution". We can replace these four words with just 
	"distribut" via a preprocessing step called **stemming**.
	There are numerous stemming strategies, some more aggressive than others. Let's use one from 
	NLTK called the `Porter stemmer`.""")
	st.markdown("""
	However, stemming can be crude and chop off suffixes haphazhardly. A better alternative is 
	**lemmatization**. For example, the word `worse` reduces to `bad`.""")
	st.markdown('The example text has now become :')
	st.write("""**`congratl numbr ticket hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send messag emailaddr get ticket`**""")
	#st.write("""**`congratlations numbr ticket hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send message emailaddr get ticket`**""")
	st.write('')
	st.write('')
	
	st.markdown("""
	What a change! Now that we've performed all operations on every data and enriched the corpus 
	with meaningful terms, we're ready to move on to *feature engineering*.""")
#app(5)