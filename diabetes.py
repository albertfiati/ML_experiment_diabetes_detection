
#ignore warnings
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#import other files
from header import *
import feature_selection_experiment as fse 
import model as dm

def loadData():
	#load dataset
	names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	dataframe = pd.read_csv('pima-indians-diabetes.data.csv', names=names)
	return dataframe

def prepareData():
	np.set_printoptions(precision=3)
	dataframe = loadData();
	print(dataframe.head(5));

	return dataframe.values;

def main():
	data = prepareData()
	X = data[:,:8]
	Y = data[:,8]

	fse.runExperiment(X, Y)

	test_size = 0.33
	seed = 7

	x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
	
	# dm.train(x_train, y_train)
	# dm.test(x_test, y_test)
	#dm.predict(x_test[0])
	

if __name__ == '__main__':
	main()

#