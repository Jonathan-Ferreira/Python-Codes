import pandas as pd
import os
import pickle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

FILE = 'census_concatenated.pkl'
CSV_FILE = 'results_census.csv'

path = os.path.abspath(FILE)

with open('census_concatenated.pkl', 'rb') as f:
    X_census, y_census = pickle.load(f)


results_grid_search = pd.DataFrame(columns=['Algorithm', 'Best Parameters', 'Best Score'])

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

results_grid_search = pd.concat([results_grid_search, new_row], ignore_index=True)


print('DecisionTreeClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

parameters = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_

new_row = pd.DataFrame({'Algorithm': ['RandomForestClassifier'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat([results_grid_search, new_row], ignore_index=True)

print('RandomForestClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

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

results_grid_search = pd.concat([results_grid_search, new_row], ignore_index=True)

print('KNeighborsClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

parameters = {'tol': [0.00001, 0.000001, 0.0000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga', 'newton-cg', 'liblinear']}

grid_search = GridSearchCV(
    estimator=LogisticRegression(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_
new_row = pd.DataFrame({'Algorithm': ['LogisticRegression'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat([results_grid_search, new_row], ignore_index=True)

print('LogisticRegression', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

parameters = {'tol': [0.00001, 0.000001, 0.0000001],
              'C': [1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

grid_search = GridSearchCV(
    estimator=svm.SVC(), param_grid=parameters)
grid_search.fit(X_census, y_census)
best_params = grid_search.best_params_
best_result = grid_search.best_score_

new_row = pd.DataFrame({'Algorithm': ['SVC'],
                        'Best Parameters': [best_params],
                        'Best Score': [best_result]})

results_grid_search = pd.concat([results_grid_search, new_row], ignore_index=True)

print('SVC', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

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

results_grid_search = pd.concat([results_grid_search, new_row], ignore_index=True)

print('MLPClassifier', '\nBest Parameters: ',
      best_params, '\nBest Result: ', best_result, '\n')

print(results_grid_search)
results_tree = []
results_randomforest = []
results_knn = []
results_logistic = []
results_SVM = []
results_neural = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    print('\n', 'Iteration number:', i+1 ,'\n')
    tree = DecisionTreeClassifier(
        criterion='gini', min_samples_leaf=1, min_samples_split=2, splitter='best')
    scores = cross_val_score(tree, X_census, y_census, cv=kfold)
    results_tree.append(scores.mean())
    print('Decision Tree: ' ,results_tree[i])

    randomforest = RandomForestClassifier(
        criterion='gini', min_samples_leaf=1, min_samples_split=2, n_estimators=150)
    scores = cross_val_score(randomforest, X_census, y_census, cv=kfold)
    results_randomforest.append(scores.mean())
    print('Random Forest: ' ,results_randomforest[i])
    
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    scores = cross_val_score(knn, X_census, y_census, cv=kfold)
    results_knn.append(scores.mean())
    print('K Neighbours: ' ,results_knn[i])

    logistic = LogisticRegression(C=1.0, solver='liblinear', tol=1e-05)
    scores = cross_val_score(logistic, X_census, y_census, cv=kfold)
    results_logistic.append(scores.mean())
    print('Logistic Regression: ' ,results_logistic[i])

    test_svm = svm.SVC(C=1.5, kernel='rbf', tol=1e-05)
    scores = cross_val_score(test_svm, X_census, y_census, cv=kfold)
    results_SVM.append(scores.mean())
    print('SVM: ' ,results_SVM[i])

    neural_network = MLPClassifier(
        activation='relu', batch_size=10, max_iter=1200, solver='adam')
    scores = cross_val_score(neural_network, X_census, y_census, cv=kfold)
    results_neural.append(scores.mean())
    print('Neural Network: ' ,results_neural[i])


results = pd.DataFrame({'Arvore': results_tree,
                        'Random Forest': results_randomforest,
                        'KNN': results_knn,
                        'Logistic Regression': results_logistic,
                        'SVM': results_SVM,
                        'Neural Network': results_neural})

print(results)

column_averages = results.mean(numeric_only=True)
print(column_averages)
top_results = column_averages.nlargest(3)
print(top_results)

results.to_csv(os.path.abspath(CSV_FILE),sep=';',index=False)

print('Final Results' ,'\n')
print('Árvore: ' ,results_tree,'\n')
print('Random Forest: ' ,results_randomforest,'\n')
print('KNN: ' ,results_knn,'\n')
print('Logistic Regression: ' ,results_logistic,'\n')
print('SVM: ' ,results_SVM,'\n')
print('Neural Network: ' ,results_neural,'\n')
