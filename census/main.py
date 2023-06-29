import data_processing as dp

chosen_columns = [1, 3, 5, 6, 7, 8, 9, 13]
categorical_list = [i for i in range(0, 14)]
target_var = 14

df = dp.generate_data('census.csv')

X_census, y_census = dp.encode(df, categorical_list, target_var)

X_census_scaled = dp.scaler(X_census)
X_census_encoded = dp.one_hot_encoder(X_census, chosen_columns)


X_census_train, y_census_train, X_census_test, y_census_test = dp.best_sampler(
    X_census_scaled, y_census, 0.15)

dp.save_result(X_census_train, y_census_train,
               X_census_test, y_census_test, 'census')
