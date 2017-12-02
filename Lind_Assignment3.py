# Illustrates a random forest classifier on the churn data.

import pandas as pd
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from data_util import *

data_a = pd.read_csv('./data/churn_data.csv')


# Get a list of the categorical features for a given dataframe.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
# Transform the data to a one-hot encoding.
data_a = pd.get_dummies(data_a, columns=cat_features(data_a))

#print the head of the data to make sure the transform was done correctly
print(data_a.head())

# Select x and y data
features = list(data_a)
#This removes customer ID from the data set because that would not be reasonable predictor of customer churn. 
features.remove('CustID')
features.remove('Churn_Yes')
#This is to take the churn no out of the data set. 
features.remove('Churn_No')

#makes set features the predictive variables. 
data_a_x = data_a[features]
#Makes churn the explanatory variable 
data_a_y = data_a['Churn_Yes']

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_a_x, data_a_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)
		
		
print('________________________________________________________________________________')

data_b = pd.read_csv('./data/churn_validation.csv')


# Get a list of the categorical features for a given dataframe.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
# Transform the data to a one-hot encoding.
data_b = pd.get_dummies(data_b, columns=cat_features(data_b))

#print the head of the data to make sure the transform was done correctly
print(data_b.head())

# Select x and y data
features = list(data_b)
#This removes customer ID from the data set because that would not be reasonable predictor of customer churn. 
features.remove('CustID')
features.remove('Churn_Yes')
#This is to take the churn no out of the data set. 
features.remove('Churn_No')
#makes set features the predictive variables. 
data_b_x = data_b[features]
#Makes churn the explanatory variable 
data_b_y = data_b['Churn_Yes']

# Make predictions - both class labels and predicted probabilities.
preds_b = mod.predict(x_test)
print("-------------Below are the predications for the validation set B---------------") 
print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
# Look at results.
print_multiclass_classif_error_report(y_test, preds_b)
