import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('census.pkl', 'rb') as f:
    X_census_train, y_census_train, X_census_test, y_census_test = pickle.load(f)

print(X_census_train.shape, y_census_train.shape)
print(X_census_test.shape, y_census_test.shape)

from sklearn import svm


linear = svm.SVC(kernel='linear', C=1).fit(X_census_train,y_census_train)
print(linear.score(X_census_test,y_census_test))

poly = svm.SVC(kernel='poly', C=1).fit(X_census_train,y_census_train)
print(poly.score(X_census_test,y_census_test))

rbf = svm.SVC(kernel='rbf', C=1).fit(X_census_train,y_census_train)
print(rbf.score(X_census_test,y_census_test))

sigmoid = svm.SVC(kernel='sigmoid', C=1).fit(X_census_train,y_census_train)
print(sigmoid.score(X_census_test,y_census_test))
# from sklearn.metrics import accuracy_score, classification_report
# print(accuracy_score(y_census_test, predictions))
# print(classification_report(y_census_test, predictions))