import pandas as pd
import numpy as np

import neural_network_helper as nnh

train_delv_path = './input/jio/TrainDeliveries.csv'
delv_data = pd.read_csv(train_delv_path)

train_match_path = './input/jio/Trainmatches.csv'
match_data = pd.read_csv(train_match_path)

columns =["id","team1","team2","winner","venue","toss_winner"]
X = match_data[columns]
t_y = pd.DataFrame(columns=['result'])
for index,row in X.iterrows():
    t_y.loc[index] = row['winner'] == row['team1']

X=pd.concat([X,t_y],axis=1)

X['team1'] = X['team1'].apply(lambda x:x.replace("Team",""))
X['team2'] = X['team2'].apply(lambda x:x.replace("Team",""))
X['venue'] = X['venue'].apply(lambda x:x.replace("Stadium",""))
X['venue'] = X['venue'].apply(lambda x:x.replace("stadium",""))

X['toss_winner'] = X['toss_winner'].apply(lambda x:x.replace("Team",""))
#X['toss_decision'] = X['toss_decision'].map({'field':1,'bat':0})



X['result'] = X['result'].apply(lambda x:x* 1)
X = X.drop(columns=['winner'],axis=1)


y = X['result']
X = X.drop('result',axis=1)


print(X.head())


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_absolute_error

#Define model
forest_model = LogisticRegressionCV(max_iter=1000)
# Fit model
forest_model.fit(train_X, train_y)

from sklearn.metrics import mean_absolute_error
yhat = forest_model.predict(val_X)
print("mean absolte error after logistic regression {}".format(mean_absolute_error(val_y, yhat)))

###----------------------

train_X['team1'] = train_X['team1'].astype(str).astype(int)

train_X['team2'] = train_X['team2'].astype(str).astype(int)
train_X['venue'] = train_X['venue'].astype(str).astype(int)
train_X['toss_winner'] = train_X['toss_winner'].astype(str).astype(int)

X_values = train_X.values;
y_data_frame = pd.DataFrame(train_y)

y_values = y_data_frame.values;

# Build a model with a n_h-dimensional hidden layer
feature_weights = nnh.nn_model_noisy_circles(X_values.T, y_values.T, n_h = 5, num_iterations = 5, print_cost=True)

# Print accuracy
predictions = nnh.predict_noisy_circles(feature_weights,X_values.T )
print ('Accuracy: %d' % float((np.dot(y_values.T,predictions.T) + np.dot(1-y_values.T,1-predictions.T))/float(y_values.T.size)*100) + '%')