import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def score_dataset(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictedY = model.predict(X_test)
    return mean_absolute_error(y_test, predictedY)

housingdata = pd.read_csv('./input/kc_house_data.csv')
# print(housingdata.describe())
# print(housingdata.columns)
# print(type(housingdata))

# Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
#        'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
#        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
#        'lat', 'long', 'sqft_living15', 'sqft_lot15'],
#       dtype='object')

# following columns will not have much impact on price
# id, zipcode, sqft_living15, sqft_lot15
# print(housingdata.head())
housingdataX = housingdata.drop(['id','price'], axis=1)
housingdataX['date'] = housingdataX['date'].apply(lambda x: x.replace('T000000',''))
print(housingdataX.head())
priceY = housingdata.price
# split the data between training and test examples
X_train, X_test, y_train, y_test = train_test_split(housingdataX, priceY, test_size=0.33, random_state=42)

# got following error
# ValueError: could not convert string to float: '20150325T000000'
# now i have to find the row
# print(X_train.loc[X_train['date'] == '20150325T000000'])

print(score_dataset(X_train, y_train, X_test, y_test))