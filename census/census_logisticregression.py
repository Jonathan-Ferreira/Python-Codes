##############################################################################################################

# WARNING: NEED TO FIX DATA, I STILL DON'T KNOW HOW. I'M WORKING ON IT

##############################################################################################################


from sklearn.linear_model import LogisticRegression
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

parameters = {'tol': [0.00001, 0.000001, 0.0000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga', 'newton-cg', 'liblinear'],
              'max_iter': [10000]}

grid_search = GridSearchCV(
    estimator=LogisticRegression(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_
new_row = pd.DataFrame({'Algorithm': ['LogisticRegression'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat(
    [results_grid_search, new_row], ignore_index=True)

print('LogisticRegression', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

print(results_grid_search)

results_logistic = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    print('\n', 'Iteration number:', i+1)
    logistic = LogisticRegression(C=1.0, solver='liblinear', tol=1e-05)
    scores = cross_val_score(logistic, X_census, y_census, cv=kfold)
    results_logistic.append(scores.mean())

print(results_logistic)
