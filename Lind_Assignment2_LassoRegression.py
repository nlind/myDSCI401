# This illustrates how to use Lasso regression to handle cases with correlated predictors.
# Lasso minimizes the Objective = RSS + alpha * (sum of absolute value of coefficients)
# When alpha = 0 this is equivalent to ordinary least squsres. 
# By adjusting alpha (> 0) we can try differing coefficient penalties.


import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import neighbors
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#To import the dataset and get rid of the N/A results. 
housing = pd.read_csv('./data/AmesHousingSetA.csv', keep_default_na=False, na_values=[""])

#To deal with the categorical features and turn them into numbers. 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. Move to util file for future use!	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
# Remove the 'PID' column - we don't want it.
del housing['PID']

# Transform the df to a one-hot encoding.
housing = pd.get_dummies(housing, columns=cat_features(housing))

#print(housing.head())

# Get all non-SalesPrice columns as features.
features = list(housing)
#remove SalesPrice as that is our response variable. 
features.remove('SalePrice')

# Get all non-SalesPrice columns and use as features/predictors.
data_x = housing[features] 

# Get SalesPrice column and use as response variable.
data_y = housing['SalePrice']

# ---------------- Part 0: Do a pairs plot to see potential relationships -------
#pd.plotting.scatter_matrix(housing, diagonal='kde')
#plt.tight_layout()
#plt.show()

# Split data into  train/test sets
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)


# ---------------- Part 1: Compare OLS vs. Lasso Regression -----------------------

# Fit the model.
# Create a least squares linear regression model.
base_mod = linear_model.LinearRegression()
base_mod.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = base_mod.predict(x_test)
print('R^2 (Base Model): ' + str(r2_score(y_test, preds)))

# Show Lasso regression fits for different alphas.
alphas = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
for a in alphas:
	# Normalizing transforms all variables to number of standard deviations away from mean.
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train, y_train)
	preds = lasso_mod.predict(x_test)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test, preds)))
