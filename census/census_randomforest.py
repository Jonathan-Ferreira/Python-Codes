from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import os
import pandas as pd
import pickle

FILE = 'census_concatenated.pkl'
CSV_FILE = 'results_census.csv'

path = os.path.abspath(FILE)

with open('census_concatenated.pkl', 'rb') as f:
    X_census, y_census = pickle.load(f)

results_grid_search = pd.DataFrame(
    columns=['Algorithm', 'Best Parameters', 'Best Score'])

parameters = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(), param_grid=parameters, verbose=1)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_

new_row = pd.DataFrame({'Algorithm': ['RandomForestClassifier'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat(
    [results_grid_search, new_row], ignore_index=True)

print('RandomForestClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

results_randomforest = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    randomforest = RandomForestClassifier(
        criterion='gini', min_samples_leaf=1, min_samples_split=2, n_estimators=150)
    scores = cross_val_score(randomforest, X_census, y_census, cv=kfold)
    results_randomforest.append(scores.mean())
    # print('Random Forest: ', results_randomforest[i])

print('Results Random Forest: ', results_randomforest)
