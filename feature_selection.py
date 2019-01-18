# import libraries
import pandas as pd 
import numpy as np 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#Filter method using chiSquare
def chiSquare(attributes, targets, no_of_atts):
	kBest = SelectKBest(score_func=chi2, k=no_of_atts)
	fit = kBest.fit(attributes, targets)
	new_feature_vector = fit.transform(attributes)
	print(fit.scores_)
	return new_feature_vector

#wrapper method using Recursive Feature Elimination
def rfe(attributes, targets, no_of_atts, model):
	rfe = RFE(model, no_of_atts)
	fit = rfe.fit(attributes, targets)
	new_feature_vector = fit.transform(attributes)
	print("Feature Ranking: %s" % (fit.ranking_))
	return new_feature_vector

#wrapper method using Principal Component Analysis
def pca(attributes, targets, no_of_atts):
	pca = PCA(no_of_atts)
	fit = pca.fit(attributes, targets)
	new_feature_vector = fit.transform(attributes)
	print("Explained variance ratio: %s" % (fit.explained_variance_ratio_))
	return new_feature_vector