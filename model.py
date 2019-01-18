from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

model_filename = "diabetes_LDA.sav"

def train(attributes_train, targets_train):
	print('Training started')
	
	model = LinearDiscriminantAnalysis()
	model.fit(attributes_train, targets_train)

	pickle.dump(model, open(model_filename,'wb'))
	
	print('Training completed')


def test(attributes_test, targets_test):
	print('Testing started')
	
	model = pickle.load(open(model_filename, 'rb'))
	result = model.score(attributes_test, targets_test)
	
	print('Testing completed')
	
	print(result)

def predict(attributes):
	print('Prediction started')

	model = pickle.load(open(model_filename, 'rb'))
	result = model.predict([attributes])
	
	print('Prediction completed')
	
	if int(result[0]) == 1:
		print('Diabetic')
	else:
		print('Not diabetic')