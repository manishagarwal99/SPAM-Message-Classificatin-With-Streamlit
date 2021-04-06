import streamlit as st

#st. set_page_config(layout="wide")

def app(n=5):
	st.header('Feature engineering')
	st.markdown("""Now that we've enriched the corpus for meaningful terms, 
	we're ready to construct features. Let's begin by breaking apart the corpus 
	into a vocabulary of unique terms&mdash;a process called **tokenization**. 
	However, there are several ways to approach this step. """)

	st.markdown("### 1. The Bag-of-Words Representation")
	st.markdown(""" It is evident that a message, considered as a sequence
	of symbols, cannot be used directly as input of a machine
	learning algorithm. Indeed, operations inside such algorithms
	require algebraic quantities, i.e. real- or complex-valued vectors
	in a given dimensional space.
	The Bag-of-Words (BoW) representation, also called
	vector space model, is a well known representation in natural
	language processing, which describes the occurrence of words
	within a given document.
	Formally, we consider a document composed of D words
	indexed as d_i = 1....D. We also consider a known
	vocabulary V which contains M reference words, where M can
	be very large. From the vocabulary, we build the representation
	associated with the document as follows:""")

	st.latex(r'''x =\sum_{i=1}^{D} x_{d_i}''')

	st.markdown("""where the result is a [one-hot vector](https://en.wikipedia.org/wiki/One-hot) in which all entries are 
	zero except the single entry corresponding to the location of
	the word d_i in the vocabulary.
	In the specific case of our dataset of SMS, we build
	the vocabulary based on all the messages of the considered
	training set.""")

	st.markdown("### 2. Tokenization")
	st.markdown(""" We can tokenize individual terms and generate 
	[bag of words model](https://en.wikipedia.org/wiki/Bag-of-words_model). 
	You may notice this model has a glaring pitfall: it fails to capture the innate 
	structure of human language. Under this model, the following sentences have 
	the same feature vector although they convey dramatically different meanings.""")
	st.markdown("""
	* `Does steak taste delicious?`
	* `Steak does taste delicious.`
	""")
	st.markdown("""
	Alternatively, we can tokenize every sequence of n terms called n-grams. 
	or example, tokenizing adjacent pairs of words yields bigrams. 
	The **n-gram model** preserves word order and can potentially capture more 
	information than the bag of words model. 

	To get the best of both worlds, let's tokenize unigrams and bigrams. 
	As an example, unigrams and bigrams for `"The quick brown fox"` are `"The"`, 
	`"quick"`, `"brown"`, `"fox"`, `"The quick"`, `"quick brown"` and `"brown fox"`.""")

	st.markdown("""### 3. Normalization of the BoW Representation: Implementing the tf-idf statistic""")
	st.markdown("""Having selected a tokenization strategy, it is preferable to 
	work with frequencies rather than occurrences.
	It is also very important have normalized features since
	methods based on distances such as k-NN classifiers may be
	biased with unnormalized feature vectors.
	A common way to perform such a task is based on the termfrequency
	(tf) inverse-document-frequency (idf) weighting,
	where the term frequency is calculated for the document n as
	follows:""")
	st.latex(r'''tf_n =\left(\frac{1}{D_n}\right)''')
	st.markdown("""where Dn designates the number of words. The tf weighting
	amounts to transforming occurrences in frequencies inside
	each document.
	It can be seen that the tf weighting does not hold any
	information to assess relevancy on a message. Indeed, words
	that appear very often among the messages have little or no
	discriminating power in determining whether a message is
	a spam or not. To address this problem, we introduce the
	document frequency dfdi of a term di as the number of
	documents in the collection that contain the term di . The idf
	is then defined as follows:""")
	st.latex(r'''idf_{d_i} =log \left(\frac{N}{df_{d_i}}\right)''')
	st.markdown("""from which we can deduce the tf-idf weighting coefficient for
	term di in document n as follows:""")
	st.latex(r'''T(n,d_i) =tf_n * idf_{d_i}''')
	st.write("""where, the matrix of occurrences weighted by the tf-idf coefficient
	is used as input of the proposed classifiers.""")
	st.markdown("""The implementation of tf-idf statistic would require following process:""")
	st.markdown("""
	* Count how many times does a word occur in each message(term frequency)
	* Weight the counts, so that frequent tokens get lower weight (inverse document frequency)
	* Normalize the vectors to unit length, to abstract from original text Length (L2 norm)
	""")
	st.markdown("""Scikit-learn has an off-the-shelf tool called `TfidfVectorizer` that performs 
	n-gram tokenization and also computes the tf-idf statistic. Two technical 
	details regarding *TfidfVectorizer* are:
	""")
	st.markdown("""
	* The tf-idf statistic is computed 
	[slightly differently](http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) 
	to avoid division by zero, and 
	* The computed tf-idf values for each training 
	example are subsequently 
	[normalized](http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).
	""")
	st.markdown("""Finally, we're equipped to transform a corpus of text data into a matrix of 
	numbers with one row per training example and one column per $n$-gram.""")

	st.markdown("""Let's take a look at the dimensions of the X_ngrams matrix formed 
	after fitting the vectorizer to the messages.""")
	st.write('`(5572, 36348)`')
	st.write('')
	st.markdown("""As we can see, that's one massive matrix! It looks like the 
	tokenization process extracted 36,348 unigrams and bigrams from the corpus; 
	each one defines a feature. Since each training example only makes use of a 
	tiny fraction of the complete  𝑛 -gram vocabulary, X_ngrams mostly consists 
	of zeros and is called a sparse matrix.
	To perform linear algebra computations rapidly on such a large sparse matrix, 
	it'd be more efficient to store only the non-zero values while maintaining the 
	structure. Fortunately, TfidfVectorizer utilizes the SciPy library to do exactly 
	this!""")

#app(5)
