from sklearn.neural_network import MLPClassifier
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

parameters = {'activation': ['relu', 'logistic', 'tanh'],
              'solver': ['adam', 'sgd'],
              'batch_size': [10, 30, 56],
              'max_iter': [1000, 1200, 1500]}

grid_search = GridSearchCV(
    estimator=MLPClassifier(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_

new_row = pd.DataFrame({'Algorithm': ['MLPClassifier'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat(
    [results_grid_search, new_row], ignore_index=True)

print('MLPClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

results_neural = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    print('\n', 'Iteration number:', i+1)

    neural_network = MLPClassifier(
        activation='relu', batch_size=10, max_iter=1200, solver='adam')
    scores = cross_val_score(neural_network, X_census, y_census, cv=kfold)
    results_neural.append(scores.mean())

print('Neural Network: ', results_neural)
