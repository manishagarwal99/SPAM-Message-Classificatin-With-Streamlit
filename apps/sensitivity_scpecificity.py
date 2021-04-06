import streamlit as st
from PIL import Image
#st. set_page_config(layout="wide")

def app(n=5):
	st.header('Sensitivity and Specificity')

	st.markdown("### 1. Increasing the Sensitivity of the Classifiers")
	st.markdown(""" The class imbalance coupled with the minimization of the
	misclassification error induce significant differences in the
	sensitivity and specificity as it can be seen on Figure 4.
	Indeed, the misclassification error is a symmetric loss. Hence
	classifiers focus more on the prediction accuracy of the most
	common class, which often results in poor accuracy for the
	other class. In some applications, one may want to increase
	the sensitivity of the classifiers. That may be the case if the
	user really wants to avoid spam.
	To increase the sensitivity, one may tune the threshold of
	the Bayes classifier associated with methods that compute posterior
	probabilities, e.g. logistic regression, neural networks,
	linear discriminant analysis etc. A higher threshold results in
	a higher specificity and a lower one to a higher sensitivity.
	In this study, we try an alternative based on resampling
	methods where the idea is to artificially balance the classes.""")

	st.markdown("### 2. Downsampling of the majority class")
	st.markdown("""We downsample the majority by randomly selecting 
	some of the entries, in such a way that we have a perfectly balanced dataset.
	Intuitively, it may not be ideal since it
	impacts the predictive performance of the classifiers;""")

	image1 = Image.open('downsampling/down_dt.png')
	image2 = Image.open('downsampling/down_knn.png')
	image3 = Image.open('downsampling/down_lr.png')
	image4 = Image.open('downsampling/down_lrl2.png')
	image5 = Image.open('downsampling/down_mnb.png')
	image6 = Image.open('downsampling/down_svc_lin.png')
	image7 = Image.open('downsampling/down_svc_rbf.png')
	image8 = Image.open('downsampling/down_svc_sig.png')
	row1, row2, row3, row4 = st.beta_columns(4)
	row1.image(image1,width=350)
	row2.image(image2,width=350)
	row3.image(image3,width=350)
	row4.image(image4,width=350)
	row1.image(image5,width=350)
	row2.image(image6,width=350)
	row3.image(image7,width=350)
	row4.image(image8,width=350)

	row = st.beta_columns((1,2.5,1))
	row[1].markdown('Fig. 5. Confusion matrices of the proposed classifiers fitted on an downsampled training set.')

	st.markdown("### 3. Upsampling of the minority class")
	st.markdown("""We randomly (with replacement) select elements of 
	the minority class that we duplicate until the two classes have the same number of elements.
	Intuitively, it should be better than downsampling
	but may lead to overfitting in case of highly
	imbalanced datasets.""")

	image1 = Image.open('upsampling/up_dt.png')
	image2 = Image.open('upsampling/up_knn.png')
	image3 = Image.open('upsampling/up_lr.png')
	image4 = Image.open('upsampling/up_lrl2.png')
	image5 = Image.open('upsampling/up_mnb.png')
	image6 = Image.open('upsampling/up_svc_lin.png')
	image7 = Image.open('upsampling/up_svc_rbf.png')
	image8 = Image.open('upsampling/up_svc_sig.png')
	row1, row2, row3, row4 = st.beta_columns(4)
	row1.image(image1,width=350)
	row2.image(image2,width=350)
	row3.image(image3,width=350)
	row4.image(image4,width=350)
	row1.image(image5,width=350)
	row2.image(image6,width=350)
	row3.image(image7,width=350)
	row4.image(image8,width=350)

	row = st.beta_columns((1,2.5,1))
	row[1].markdown('Fig. 6. Confusion matrices of the proposed classifiers fitted on an upsampled training set.')

	st.markdown("### 4. Results")
	st.markdown("""Figures 5 and 6 display the confusion matrices of the
	proposed classifiers associated with an upsampled and a downsampled
	training set, respectively.
	First, it can be noticed that the downsampled dataset leads
	to lower predictive performance as expected, since all the
	methods suffer from a higher misclassification error than
	with a normal dataset. However, the sensitivity has been
	significantly increased, by more than several percent for all
	the methods. k-NN even gets a 30% increase of its sensitivity.
	But evidently, the price to pay is a lower specificity.
	Regarding the upsampled training set, the performance
	of the classifiers are slightly better than with the normal
	dataset. Such an increase can be explained by a slightly higher
	sensitivity. However, it can be observed that k-NN may overfit
	resulting in a lower sensitivity.""")

	st.write('')
	st.write('')

	st.markdown("""
	Thus, it is clear that upsampling is dangerous in terms of
	overfitting and has a relatively small impact on the sensitivity,
	while downsampling works significantly better on the sensitivity
	but is sub-optimal in terms of misclassification error.""")

#app(5)
