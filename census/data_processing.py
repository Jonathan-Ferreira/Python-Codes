from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
import pickle as pkl
import pandas as pd
import numpy as np


# I created this function to auto-generate the original data for ease of testing.
def generate_data(file_name):
    path = os.path.abspath(file_name)
    # Reading CSV file
    df = pd.read_csv(path, sep=',')

    data_frame = pd.DataFrame(df)
    return data_frame


def encode(data_frame, categorical_var_list, target_var):
    # Attributing values to X and Y for machine learning training
    # X represents the independent features, which means the types of classifications you want to analyse
    # Y represents the target variable

    # in this part of the code: [:, 0:14]
    # "[:" indicates you want all the lines in the dataset, while ",0:14]" indicates you're attributing the columns from index 0 to 13
    X_variable = data_frame.iloc[:, categorical_var_list].values
    y_variable = data_frame.iloc[:, target_var].values

    label_encoder_workclass = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_marital = LabelEncoder()
    label_encoder_occupation = LabelEncoder()
    label_encoder_relationship = LabelEncoder()
    label_encoder_race = LabelEncoder()
    label_encoder_sex = LabelEncoder()
    label_encoder_country = LabelEncoder()

    X_variable[:, 1] = label_encoder_workclass.fit_transform(X_variable[:, 1])
    X_variable[:, 3] = label_encoder_education.fit_transform(X_variable[:, 3])
    X_variable[:, 5] = label_encoder_marital.fit_transform(X_variable[:, 5])
    X_variable[:, 6] = label_encoder_occupation.fit_transform(X_variable[:, 6])
    X_variable[:, 7] = label_encoder_relationship.fit_transform(
        X_variable[:, 7])
    X_variable[:, 8] = label_encoder_race.fit_transform(X_variable[:, 8])
    X_variable[:, 9] = label_encoder_sex.fit_transform(X_variable[:, 9])
    X_variable[:, 13] = label_encoder_country.fit_transform(X_variable[:, 13])

    return X_variable, y_variable


def data_split(X_values, y_values, test_size):
    X_variable_train, X_variable_test, y_variable_train, y_variable_test = train_test_split(
        X_values, y_values, test_size=test_size, random_state=0)
    print('Size of the training values: \n ', 'Independent values:',
          X_variable_train.shape, '\n  Target values: ', y_variable_train.shape, '\n')
    print('Size of the testing values: \n ', 'Independent values:',
          X_variable_test.shape, '\n  Target values: ', y_variable_test.shape, '\n')

    return X_variable_train, X_variable_test, y_variable_train, y_variable_test


# This function normalizes the values of the column so they operate in the same scales to avoid biases during machine learning operations
def scaler(X_variable):
    scaler = StandardScaler()
    X_variable_scaled = scaler.fit_transform(X_variable)
    return X_variable_scaled


# This function turns categorical variables into numerical
def one_hot_encoder(X_variable, chosen_columns):
    if type(chosen_columns) is list:
        onehotencoder = ColumnTransformer(transformers=[(
            'OneHot', OneHotEncoder(), chosen_columns)], remainder='passthrough')
        X_variable_encoded = onehotencoder.fit_transform(
            X_variable).toarray()
        return X_variable_encoded
    else:
        raise TypeError("Chosen columns must be a list")


def _sampling_tomek(X_variable, y_variable, sample_split):
    ############################################################################
    #        Doing a Tomek Link to under-sample data
    ############################################################################

    tl = TomekLinks(sampling_strategy='majority')
    X_under, y_under = tl.fit_resample(X_variable, y_variable)

    X_variable_train_under, X_variable_test_under, y_variable_train_under, y_variable_test_under = data_split(
        X_under, y_under, test_size=sample_split)

    return X_variable_train_under, y_variable_train_under, X_variable_test_under, y_variable_test_under


def _sampling_SMOTE(X_variable, y_variable, sample_split):
    ############################################################################
    #        Doing a SMOTE sampling method to over-sample data
    ############################################################################

    smt = SMOTE(sampling_strategy='minority')
    X_over, y_over = smt.fit_resample(X_variable, y_variable)

    X_variable_train_over, X_variable_test_over, y_variable_train_over, y_variable_test_over = data_split(
        X_over, y_over, test_size=sample_split)

    return X_variable_train_over, y_variable_train_over, X_variable_test_over, y_variable_test_over


def _test_sampling(X_variable_train, y_variable_train, X_variable_test, y_variable_test):
    random_forest = RandomForestClassifier(
        criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    random_forest.fit(X_variable_train, y_variable_train)

    predictions = random_forest.predict(X_variable_test)
    accuracy = accuracy_score(y_variable_test, predictions)
    return accuracy


def best_sampler(X_variable, y_variable, sample_split):
    if (type(sample_split) is not float) and (type(sample_split) is not int):
        raise ValueError("Invalid value")
    X_tomek_train, y_tomek_train, X_tomek_test, y_tomek_test = _sampling_tomek(
        X_variable, y_variable, sample_split)

    accuracy_tomek = _test_sampling(
        X_tomek_train, y_tomek_train, X_tomek_test, y_tomek_test)

    X_smote_train, y_smote_train, X_smote_test, y_smote_test = _sampling_SMOTE(
        X_variable, y_variable, sample_split)
    accuracy_SMOTE = _test_sampling(
        X_smote_train, y_smote_train, X_smote_test, y_smote_test)

    if accuracy_tomek > accuracy_SMOTE:
        print("Tomek sampling is more precise")
        X_variable_train = X_tomek_train
        y_variable_train = y_tomek_train
        X_variable_test = X_tomek_test
        y_variable_test = y_tomek_test
    else:
        print("SMOTE sampling is more precise")
        X_variable_train = X_smote_train
        y_variable_train = y_smote_train
        X_variable_test = X_smote_test
        y_variable_test = y_smote_test
    print()

    return X_variable_train, y_variable_train, X_variable_test, y_variable_test


def save_result(X_variable_train, y_variable_train, X_variable_test, y_variable_test, file_name):
    full_name = file_name + '.pkl'
    with open(full_name, mode='wb') as f:
        pkl.dump([X_variable_train, y_variable_train,
                  X_variable_test, y_variable_test], f)
    print('File saved on:', os.path.abspath(full_name))

    X_variable_concatenated = np.concatenate(
        (X_variable_train, X_variable_test), axis=0)
    y_variable_concatenated = np.concatenate(
        (y_variable_train, y_variable_test), axis=0)

    concat_name = file_name + '_concatenated.pkl'
    with open(concat_name, mode='wb') as f:
        pkl.dump([X_variable_concatenated, y_variable_concatenated], f)
    print('File saved on:', os.path.abspath(concat_name))
