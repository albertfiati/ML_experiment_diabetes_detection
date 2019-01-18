# load libraries needed for the example
import pandas	
import sklearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import model_selection
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np

# compute f-measure for the model
def fmeasure(precision, recall):
	f_measure = []
	beta = 0.5

	for i in range(len(precision)):
		f_measure.append( 1 / ((beta*(1/precision[i])) + ((1-beta)*(1/recall[i]))) )

	return f_measure

# build the models to be used for the validation
def buildModels():
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	models.append(('NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))

	return models

# format the results
def formatResults(names, accuracy_results, f_measure_results, fit_times):
	print('Accuracy')
	print('-------------------------')
	res = [{names[i]: accuracy_results[i]} for i in range(len(names))]
	print(res)
	print()

	print('F-Measure')
	print('-------------------------')
	res = [{names[i]: f_measure_results[i]} for i in range(len(names))]
	print(res)
	print()

	print('Fit time')
	print('-------------------------')
	res = [{names[i]: fit_times[i]} for i in range(len(names))]
	print(res)
	print()


# creating k-fold cross validation (k=10). This will split the data into 10 parts. Train on 9 and test on 1 part. 
def validate(attributes, targets):
	validation_size = 0.2	# define the test size 
	seed = 6				# defines a seed to be used to keep same randomness in training and testing dataset

	# perform model selection by splitting the data into train and test groups
	x_train, x_test, y_train, y_test = model_selection.train_test_split(attributes, targets, test_size=validation_size, random_state=seed)
	models = buildModels()
	

	# Evaluating each of the models
	k = 10						# define the value of k for the splits
	names = []					# hold the name of the model been trained
	recall_results = []			# hold the results of the recall evaluations
	accuracy_results = []		# hold the results of the accuracy evaluations
	precision_results = []		# hold the results of the precision evaluations
	f_measure_results = []		# hold the f-measure values
	fit_times = []				# hold the training times for the models

	# loop through all the models and collect metrics
	for name, model in models:
		# cross validation model selection
		#cv = model_selection.KFold(n_splits=k,random_state=seed)	# k-fold cross validation
		cv = model_selection.StratifiedKFold(n_splits=k)			# stratified k-fold cross validation
		
		# performance measures
		accuracy = model_selection.cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy')		# accuracy score
		precision = model_selection.cross_val_score(model, x_train, y_train, cv=cv, scoring='precision')	# precision score
		recall = model_selection.cross_val_score(model, x_train, y_train, cv=cv, scoring='recall')			# recal score
		fit_time = model_selection.cross_validate(model, x_train, y_train, cv=cv)							# collect the training time for each fold
		f_measure = fmeasure(precision, recall)																# fmeasure score

		# add respective score arrays to the global score array
		accuracy_results.append(accuracy)
		precision_results.append(precision)
		recall_results.append(recall)
		f_measure_results.append(f_measure)
		fit_times.append(fit_time.get('fit_time')	)
		names.append(name)

		msg = "%s: %f (%f)" % (name, accuracy.mean(), accuracy.std())
		print(msg)

	#return formatResults(names, accuracy_results, f_measure_results, fit_times)






 
