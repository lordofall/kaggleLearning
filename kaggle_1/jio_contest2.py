import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder
from classification_model import classification_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

train_match_path = './input/jio/Trainmatches.csv'
match_data_frame = pd.read_csv(train_match_path)

match_data_frame['team_1_win_flag'] = np.where(match_data_frame.team1 == match_data_frame.winner, 1, 0);
df = match_data_frame.drop(['id','season','dl_applied','win_by_runs','win_by_wickets','player_of_match','result', 'winner'], axis=1);
# columns in df ['team1', 'team2', 'toss_winner', 'toss_decision', 'result', 'winner', 'venue']
# print(df.columns);
# print(df.head());
# print(df.size); #3500
# print(df.shape); #500 x 7
# give me unique teams
# print(pd.unique(df.toss_winner));
# check for null values
# print(df[pd.isnull(df['winner'])]);
# df['winner'].fillna('Draw',inplace=True);
# print(df[pd.isnull(df['venue'])]);

encode = {'team1': {'Team1':1,'Team2':2,'Team3':3,'Team4':4,'Team5':5,'Team6':6,'Team7':7,'Team8':8,'Team9':9,'Team10':10,'Team11':11},
          'team2': {'Team1':1,'Team2':2,'Team3':3,'Team4':4,'Team5':5,'Team6':6,'Team7':7,'Team8':8,'Team9':9,'Team10':10,'Team11':11},
          'toss_winner': {'Team1':1,'Team2':2,'Team3':3,'Team4':4,'Team5':5,'Team6':6,'Team7':7,'Team8':8,'Team9':9,'Team10':10,'Team11':11},
          'winner': {'Team1':1,'Team2':2,'Team3':3,'Team4':4,'Team5':5,'Team6':6,'Team7':7,'Team8':8,'Team9':9,'Team10':10,'Team11':11,'Draw':12}}
df.replace(encode, inplace=True)



#we maintain a dictionary for future reference mapping teams
# dicVal = encode['winner']
# print(dicVal);

# print(df.describe());

# print(df.iloc[300]);

#Find some stats on the match winners and toss winners

'''
temp1=df['toss_winner'].value_counts(sort=True)
temp2=df['winner'].value_counts(sort=True)

print('No of toss winners by each team')
for idx, val in temp1.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
print('No of match winners by each team')
for idx, val in temp2.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
'''

'''
df['winner'].hist(bins=50);
plot.show();
'''

#find the null values in every column
# print(df.apply(lambda x: np.sum(x.isnull(), axis=0)));
# print(df.columns);

# categorical variables should be converted into numeric variables using  the concept of encoding with scikit - learn LabelEncoder
var_mod = ['city','toss_decision','venue'];
le = LabelEncoder();
for i in var_mod:
    df[i] = le.fit_transform(df[i]);
# print(df.dtypes)



model = RandomForestClassifier(n_estimators=100)
# model = LogisticRegression()
# model = LogisticRegressionCV()
outcome_var = ['team_1_win_flag']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var);

'''
#feature importances: If we ignore teams, Venue seems to be one of important factors in determining winners
#followed by toss winning, city (feature_importance is for RandomForestRegression only)
imp_input = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(imp_input)
'''
# testing

test_match_path = './input/jio/Testmatches.csv'
test_data_frame = pd.read_csv(test_match_path);
# 'match_id', 'season', 'city', 'team1', 'team2', 'toss_winner','toss_decision', 'result', 'dl_applied', 'venue'
test_df = test_data_frame.drop(['match_id', 'season', 'result', 'dl_applied'], axis=1);
test_df.replace(encode, inplace=True)
for i in var_mod:
    test_df[i] = le.fit_transform(test_df[i]);
outcome = model.predict(test_df)

my_submission = test_data_frame.drop(['season', 'city','team1', 'team2', 'toss_winner','toss_decision', 'result', 'dl_applied', 'venue'], axis=1)
my_submission['team_1_win_flag'] = pd.DataFrame(outcome)
# my_submission = pd.concat([test_data_frame.match_id, outcome_df],  axis=1 )
print(my_submission.head(5))
my_submission.to_csv('submission1.csv', index=False)