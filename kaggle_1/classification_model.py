# Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import numpy as np

# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors], data[outcome].values.ravel())

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
    kf = KFold
    kf = KFold(n_splits=2)
    error = []
    for train, test in kf.split(data):
        train_predictors = (data[predictors].iloc[train, :])

        train_target = data[outcome].iloc[train]

        model.fit(train_predictors, train_target.values.ravel())

        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))

    model.fit(data[predictors], data[outcome].values.ravel())