# This illustrates several feature selection methods for regression.

import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
import seaborn as sns 

#Get the data from excel and get rid of the N/A values. 
housing = pd.read_csv('./data/AmesHousingSetA.csv', keep_default_na=False, na_values=[""])

# Get a list of the categorical features for a given dataframe. Move to util file for future use!
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. Move to util file for future use!	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

#print(cat_feature_inds(housing))


# Remove the 'PID' column - we don't want it
del housing['PID']

# ---------------- Part 0: Do a pairs plot to see potential relationships -------
#pd.plotting.scatter_matrix(housing, diagonal='kde')
#plt.tight_layout()
#plt.show()


#make a data frame that includes attributes I think will have a large amount of influence 
#on the response variable (housing price)
df1 = housing.filter(['SalePrice', 'Year.Built', 'Lot.Area', 'Year.Remod.Add', 'Overall.Cond', 
	'Lot.Frontage', 'Overall.Qual', 'Mas.Vnr.Area', 'Total.Bsmt.SF', 'Gr.Liv.Area'], axis =1);

#I am using a seaborn's heatmap 
corr = df1.corr(); 
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values);
#ok that's much easier to understand (to read it more clearly maximize the page) 
plt.show()

# Transform the df to a one-hot encoding.
housing = pd.get_dummies(housing, columns=cat_features(housing))

print(housing.head())

# Get all non-SalesPrice% columns as features.
features = list(housing)
#remove SalesPrice because that is our response variable. 
features.remove('SalePrice')

# Get all non-SalesPrice columns and use as features/predictors.
data_x = housing[features] 

# Get SalePrice column and use as response variable.
data_y = housing['SalePrice']

# ---------------- Part 0: Plot data, split data, and build a baseline model ----

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
base_model = linear_model.LinearRegression()

# Fit the model.
base_model.fit(x_train, y_train)

# Make predictions on test data and look at the results.
preds = base_model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Base Model): ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 	

# ---------------- Part 1: Use %ile-based feature selection to build the model --

# Create a percentile-based feature selector based on the F-scores. Get top 25% best features by F-test.
selector_f = SelectPercentile(f_regression, percentile=25)
selector_f.fit(x_train, y_train)

# Print the f-scores
for name, score, pv in zip(list(housing), selector_f.scores_, selector_f.pvalues_):
	print('F-score, p-value (' + name + '): ' + str(score) + ',  ' + str(pv))

# Get the columns of the best 25% features.	
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Fit the model.
model.fit(xt_train, y_train)

# Make predictions on test data and look at the results.
preds = model.predict(xt_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Top 25% Model): ' + \
							   str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 	
						   
						   
# ---------------- Part 2: Use k-best feature selection to build the model --

# Create a top-k feature selector based on the F-scores. Get top 25% best features by F-test.
selector_f = SelectKBest(f_regression, k=3)
selector_f.fit(x_train, y_train)

# Get the columns of the best 25% features.	
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Fit the model.
model.fit(xt_train, y_train)

# Make predictions on test data and look at the results.
preds = model.predict(xt_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Top 3 Model): ' + \
							   str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 	
						   						   
												   
# ---------------- Part 3: Use Recursive Feature Elimination with Cross Validation -

# Use RFECV to arrive at the approximate best set of predictors. RFECV is a greedy method.
selector_f = RFECV(estimator=linear_model.LinearRegression(), \
                   cv=5, scoring=make_scorer(r2_score))
selector_f.fit(x_train, y_train)

# Get the columns of the best 25% features.	
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Fit the model.
model.fit(xt_train, y_train)

# Make predictions on test data and look at the results.
preds = model.predict(xt_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (RFECV Model): ' + \
							   str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 