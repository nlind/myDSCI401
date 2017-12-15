# This is a random forest classifier model for my stock return project. It should predict what 
#factors best influence a positive return for the Student Managed Investment Fund. 

#These are the imported libraries 
import pandas as pd
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from data_util import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import naive_bayes
from sklearn.model_selection import GridSearchCV

#This is how we import the data as a CSV file. Make sure the excel spreadsheet is saved under .csv. 
#returns = pd.read_csv('./data/DATA_FINALPROJECT_A.csv')#, keep_default_na=False, na_values=[""])
val_returns = pd.read_csv('./data/final_project_data.csv')#, keep_default_na=False, na_values=[""])

#_______________________________________data_tranforms_________________________________________	
# Remove the 'NAME ' column - it will effect code but we want it in the data so we know which 
#person filled out the survey and for what stock. 
#del returns['NAME ']
del val_returns['NAME ']

# Transform the df to a one-hot encoding.
#returns = pd.get_dummies(returns, columns=cat_features(returns))
val_returns = pd.get_dummies(val_returns, columns=cat_features(val_returns))

# This is a quick sanity check to make sure all of the variables were encoded correctly. 
#print(returns.head()) 
print(val_returns.head())

#This makes a list of all of our explanatory or predictor variables called "features" 
#features = list(returns)
features = list(val_returns)
#remove AVG_Return_Fluc as that is our response variable. We do not want that as one of our predictor 
#variable features. 
#features.remove('AVG_RETURN_FLUC_LL_3MONTHS')
features.remove('AVG_RETURN_FLUC')

#________________________________________Get_explanatory_and_response_variables_________________

# Get all non-returns columns and use as features/predictors (set as x variable).
#data_x = returns[list(returns)[1:]] 
Validation_data_x = val_returns[list(val_returns)[1:]] 

# Get Avg_return_fluctuations column and use as response variable.
#data_y = returns[list(returns)[0]] 
#data_y = returns['AVG_RETURN_FLUC_LL_3MONTHS']
Validation_data_y = val_returns['AVG_RETURN_FLUC']

#We print the returns as a sanity check to make sure all of the variables are still coded correctly 
#and that avg_return_fluc is not included. 
#print(returns) 
print(val_returns)

#____________________________________________Make_Model_________________________________________________
# Split training and test sets from the main set. Note: random_state just enables results to be repeated.
#x_train, x_test, y_train, y_test = train_test_split(Validation_data_x, Validation_data_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for different n_est and depth values. 
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		#mod.fit(x_train, y_train)
		mod.fit(Validation_data_x,Validation_data_y)

		# Make predictions - both class labels and predicted probabilities.
		#preds = mod.predict(x_test)
		preds = mod.predict(Validation_data_x)
		#print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		#print_multiclass_classif_error_report(y_test, preds)
		print('---------- EVALUATING Validation MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		print_multiclass_classif_error_report(Validation_data_y, preds)
		
#________________________________________Optimize_model________________________________________________
# Here is a basic voting classifier with CV and Grid Search.
m1 = svm.SVC()
m2 = ensemble.RandomForestClassifier()
m3 = naive_bayes.GaussianNB()
voting_mod = ensemble.VotingClassifier(estimators=[('svm', m1), ('rf', m2), ('nb', m3)], voting='hard')

# Set up params for combined Grid Search on the voting model. Notice the convention for specifying 
# parameters foreach of the different models.
param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]}
best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=5)
best_voting_mod.fit(Validation_data_x, Validation_data_y)
print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(Validation_data_x, Validation_data_y)))