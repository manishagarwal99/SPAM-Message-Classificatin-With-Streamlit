import streamlit as st
import pandas as pd
from PIL import Image
#st. set_page_config(layout="wide")

def app(n=5):
	st.header('Comparison of several Classifiers')

	st.markdown("### 1. Considered Classifiers")
	st.markdown(""" The goal of the project is to design a classifier which takes a feature vector containing the
	normalized word frequencies of a given message and assigns
	it to either ham (class 0) or spam (class 1).
	""")
	st.markdown("""We implement the following statistical learning methods to
	predict the class of a message:""")

	st.markdown("""
	* `k-nearest neighbors (kNN)`: kNN is a non-parametric
	approach which is highly flexible and uses non-linear
	decision boundaries. Before the implementation of kNN,
	it is crucial to scale the data since this method relies on
	the euclidean distance between observations. The number
	of neighbours is selected by cross-validation;

	* `Multinomial naive Bayes (MNB)`: The multinomial
	naive Bayes classifier models the conditional probability
	of a feature *h* given its class by a multinomial distribution.
	The additive smoothing parameter is tuned by crossvalidation;

	* `Logistic regression (LR)`: Logistic regression models the
	posterior probability of each class given a data point using
	the logistic function;

	* `L2-regularized logistic regression (LR-l2)`: Logistic regression
	with `L2 penalization on the weight vector. It is
	used when feature vectors are not sparse and to avoid
	overfitting. The regularization parameter C is tuned
	by cross-validation;

	* `Soft-margin support vector machine (SVM)`: SVM
	is a separating hyperplane classifier which models the
	boundary between classes as an hyperplane with maximal
	margin. Three kernels are tested, e.g. linear (SVM-L),
	radial basis functions (SVM-R) and sigmoid (SVM-S).
	The regularization parameter C is applied on the slack
	variables is tuned by cross-validation;

	* `Decision trees (DT)`: Decision trees stratify the feature
	space recursively into simple regions and assign the label
	ham or spam using the majority vote. The DT are fitted
	using a cost-complexity pruning based on the Gini index
	and with a minimum number of leaves estimated by crossvalidation;

	* `Random forests (RF)`: Random forests are based on
	averaging DT grown from random subset of features. The
	DT are fitted using a cost-complexity pruning based on
	the Gini index and with a minimum number of leaves
	estimated by cross-validation. Bootstrap of the samples
	are used to grow the trees;""")

	st.markdown("### 2. Experimental settings")
	st.markdown("""The dataset is divided into a training and a test set with
	a 80/20 split. Hyper-parameter tuning is achieved by 10-
	fold [cross validation](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f) 
	based on the [misclassification error](https://towardsdatascience.com/machine-learning-an-error-by-any-other-name-a7760a702c4d) on
	the training set. Once the best estimators are identified, the
	following metrics are computed on the test set:
	""")
	row1, row2 = st.beta_columns((2,1))

	image0 = Image.open('normal/confusion_matrix.png')
	row2.image(image0, caption='Confusion Matrix')

	row1.markdown("""
	* Misclassification error (ME)

	* Sensitivity (SE): The sensitivity designates the probability
	of predicting spam given that the true class is spam.

	* Specificity (SP): The specificity designates the probability
	of predicting ham given that the true class is ham.
	""")

	row1.markdown("""In this section, we assume that sensitivity and specificity are
	of equal importance in our setting. Thus, our objective is to
	minimize the total misclassification error.
	The algorithms and methods are implemented on Python,
	with scikit-learn library.""")

	st.markdown("### 3. Using nested cross-validation to minimize information leakage")
	st.markdown("""To optimize hyperparameters, we can use Scikit-learn's `GridSearchCV` tool, 
	which trains a series of candidate models using every combination of hyperparameters, 
	evaluates each model using $k$-fold cross-validation, and then reports the "winning" model and its hyperparameter combination that yielded the best performance. 
	However, we can't report this value as an unbiased estimate of the model's performance because we repeatedly reused the same data for cross-validation&mdash;we potentially "leaked" information across the candidate models!
	""")
	st.markdown("""What we can do is utilize **nested cross-validation** to alleviate this issue. In this procedure, we implement $k$-fold cross-validation to train $k$ models (the outer loop). 
	Using the *training set* of each fold, we perform `GridSearchCV` to tune the hyperparameters and select a winning model (the inner loop).
	Then, using the *validation set* of each fold, we evaluate the performance of the winning model developed in the inner loop. 
	Finally, by computing the mean of this performance value across the $k$ folds, we can report a robust estimate of the model's performance.""")
	st.markdown("""Using nested cross-validation, let's test a range of 20 values for the regularization hyperparameter and use 10-fold cross-validation to assess the classifier's performance.""")

	st.markdown("### 4. Results")
	st.markdown("""Figure displays the confusion matrices as well as the
	misclassification error for the proposed classifiers. It can be
	noticed that all the classifiers perform remarkably well with
	a maximum misclassification error of 4.7% for the k-NN
	classifiers. It can also be observed that LR-based classifiers
	as well as kernel SVM perform slightly better than
	the other classifiers, with a similar misclassification error of
	2.6 %.""")
	st.markdown("""
	The reason why tree-based methods do not perform as
	well as others may be due to the sparseness of the feature
	vectors, which makes the stratification of the feature space
	rather difficult.""")
	st.markdown("""
	The classifiers have a very high sensitivity but suffer from a
	lower specificity which may be preferable. Indeed, that means
	that nearly all the ham messages are classified as ham while
	some messages are misclassified. A deeper investigation of
	sensitivity and specificity will be achieved in later section.
	Regarding the robustness of the proposed training and
	comparison strategies, the fact that we use 10-fold crossvalidation
	makes the proposed comparison relatively robust
	to hyper-parameter changes""")

	image1 = Image.open('normal/no_dt.png')
	image2 = Image.open('normal/no_knn.png')
	image3 = Image.open('normal/no_lr.png')
	image4 = Image.open('normal/no_lrl2.png')
	image5 = Image.open('normal/no_mnb.png')
	image6 = Image.open('normal/no_svc_lin.png')
	image7 = Image.open('normal/no_svc_rbf.png')
	image8 = Image.open('normal/no_svc_sig.png')
	row1, row2, row3, row4 = st.beta_columns(4)
	row1.image(image1,width=350)
	row2.image(image2,width=350)
	row3.image(image3,width=350)
	row4.image(image4,width=350)
	#row1, row2, row3, row4 = st.beta_columns(4)
	row1.image(image5,width=350)
	row2.image(image6,width=350)
	row3.image(image7,width=350)
	row4.image(image8,width=350)

	row = st.beta_columns(3)
	row[1].markdown('Fig. 4. Confusion matrices of proposed classifiers')

	st.markdown("""### 5. What terms are the top predictors of spam?""")
	st.markdown("Now, let's take a look at the top 20 $n$-grams that are most predictive of spam.")
	df = pd.read_pickle("pickle-files/top_20.pkl")
	st.write(df)

	st.markdown("""There are a few obvious ones at the top: `phonenum`, `text`, `moneysymbnum` (money amount), `call phnnum`, `service`, `num`, `free` and `credit`. 
	""")
	st.write('')
	st.markdown("""However, the [data description](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) points out the majority of the spam in this dataset originated from a British website, while the most of the ham came from Singaporean students. 
	This is quite concerning because the lexicon in SMS messages varies dramatically across different national, cultural and age demographics. 
	This sampling bias may adversely affect the validity of the classifier.
	Of course, the biggest issue in this analysis stems from the dataset itself. We discovered the training examples weren't **independently and identically distributed**, which breaks an 
	[important assumption](https://stats.stackexchange.com/questions/213464/on-the-importance-of-the-i-i-d-assumption-in-statistical-learning) in machine learning. 
	Therefore, to improve the classifier, it's crucial to not only acquire *more* training examples but ensure they all come from the same distribution.""")

#app(5)
