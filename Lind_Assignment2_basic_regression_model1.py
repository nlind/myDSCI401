
# This illustrates a basic regression case.- this is my base model. 

import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns 

housing = pd.read_csv('./data/AmesHousingSetA.csv', keep_default_na=False, na_values=[""])

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. Move to util file for future use!	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

del housing['PID']

# Transform the df to a one-hot encoding.
housing = pd.get_dummies(housing, columns=cat_features(housing))

print(housing.head())

# Get all non-SalePrice columns as features.
features = list(housing)
features.remove('SalePrice')

# Get all non-SalePrice columns and use as features/predictors.
data_x = housing[features] 

# Get  SalePrice column and use as response variable.
data_y = housing['SalePrice']

# ---------------- Part 0: Do a pairs plot and seaborn heatmap to see potential relationships -------
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

# ---------------- Part 1: Do a basic linear regression -------------------------

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Training data used to train the algorithm, test is to assess performance.
# These should always be different - we will correct this below.
x_train, x_test, y_train, y_test = data_x, data_x, data_y, data_y


# Fit the model.
model.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

# ---------------- Part 2: Do linear regression with proper train/test split -------

# Create a least squares linear regression model.
model = linear_model.LinearRegression()

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)

# Fit the model.
model.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS, recall: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
						