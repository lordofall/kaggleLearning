import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#melbourne_file_path = './input/melb_data/melb_data.csv'
iowa_file_path='./input/train.csv'
#melbourne_data = pd.read_csv(melbourne_file_path)
iowa_data = pd.read_csv(iowa_file_path)


#print(melbourne_data.describe())
print(iowa_data.columns)
#y=melbourne_data.Price
#print(melbourne_price_data.head())

#columns_of_interest = ['Landsize', 'BuildingArea']
#two_columns_of_data = melbourne_data[columns_of_interest]
#print(two_columns_of_data.describe())

melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']

iowa_predictors = ['LotArea',
                    'YearBuilt',
                    '1stFlrSF',
                    '2ndFlrSF',
                    'FullBath',
                    'BedroomAbvGr',
                    'TotRmsAbvGrd']
X = iowa_data[iowa_predictors]
y = iowa_data.SalePrice
# X and y are divided into training data and validation data. build the model on training data
# and then validate your model with validation data.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0 )
# Define model
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()


# Fit model
decision_tree_model.fit(train_X, train_y)

# predict the outcome of validation data
predicted_home_prices = decision_tree_model.predict(val_X);

# check the mean absolute error of predicted outcome and actual outcome
error_in_predictions = mean_absolute_error(val_y, predicted_home_prices)
print(error_in_predictions)


random_forest_model.fit(train_X, train_y)
iowa_preds = random_forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))



# Read the test data
test = pd.read_csv('../input/test.csv')
print(test.columns)
# Treat the test data in the same way as training data. In this case, pull same columns.
# predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
test_X = test[iowa_predictors]
# Use the model to make predictions
predicted_prices = random_forest_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)