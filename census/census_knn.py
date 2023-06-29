from sklearn.neighbors import KNeighborsClassifier
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

parameters = {'n_neighbors': [3, 5, 10, 20],
              'p': [1, 2]}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_

new_row = pd.DataFrame({'Algorithm': ['KNeighborsClassifier'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat(
    [results_grid_search, new_row], ignore_index=True)

print('KNeighborsClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')


print(results_grid_search)

results_knn = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    print('\n', 'Iteration number:', i+1)

    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    scores = cross_val_score(knn, X_census, y_census, cv=kfold)
    results_knn.append(scores.mean())

print('KNN: ', results_knn, '\n')
