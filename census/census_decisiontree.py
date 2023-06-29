from sklearn.tree import DecisionTreeClassifier
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
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_

new_row = pd.DataFrame({'Algorithm': ['DecisionTreeClassifier'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat(
    [results_grid_search, new_row], ignore_index=True)

print('DecisionTreeClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

results_tree = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    tree = DecisionTreeClassifier(
        criterion='gini', min_samples_leaf=1, min_samples_split=2, splitter='best')
    scores = cross_val_score(tree, X_census, y_census, cv=kfold)
    results_tree.append(scores.mean())
    # print('Decision Tree: ', results_tree[i])

print('Results Decision Tree: ', results_tree)
