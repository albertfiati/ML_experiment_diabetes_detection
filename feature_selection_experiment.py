from header import *
import feature_selection as fs 
import k_fold_validation as kf

def runExperiment(X, Y):
	no_of_features = 7

	print('')
	print('Before feature selection')
	kf.validate(X, Y)
	print('')

	print('')
	print('chiSquare feature selection')
	X_new_chiSquare = fs.chiSquare(X, Y, no_of_features)
	print(X_new_chiSquare)
	print('')
	kf.validate(X_new_chiSquare, Y)
	print('')

	print('')
	print('Recursive feature Elimination feature selection - 1')
	model = LogisticRegression()
	X_new_RFE = fs.rfe(X, Y, no_of_features, model)
	print(X_new_RFE)
	print('')
	kf.validate(X_new_RFE, Y)
	print('')

	print('')
	print('Recursive feature Elimination feature selection - 2')
	model = LinearDiscriminantAnalysis()
	X_new_RFE = fs.rfe(X, Y, no_of_features, model)
	print(X_new_RFE)
	print('')
	kf.validate(X_new_RFE, Y)
	print('')

	print('')
	print('Recursive feature Elimination feature selection - 3')
	model = DecisionTreeClassifier()
	X_new_RFE = fs.rfe(X, Y, no_of_features, model)
	print(X_new_RFE)
	print('')
	kf.validate(X_new_RFE, Y)
	print('')

	print('')
	print('Principal Component Analysis feature selection')
	X_new_PCA = fs.pca(X, Y, no_of_features)
	print(X_new_PCA)
	print('')
	kf.validate(X_new_PCA, Y)
	print('')