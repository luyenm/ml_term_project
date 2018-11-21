import csv_reader as reader
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# TODO: Implement forward-stepwise subset selection.
# TODO: Implement ridge regression for the 110 columns after data pre-processing


# Performs ridge regression and K-Fold cross validation.
# x - inputs
# y - outputs
# k - kfolds
# p - lambda
def ridge_regression(training_input, training_output, k_folds, p):
    number_of_fold_indexes = int(len(training_input) / k_folds)
    train_error = []
    cv_error = []
    ridge = Ridge(alpha=p)
    for i in range(k_folds):
        index_start = int(i * number_of_fold_indexes)
        index_end = int(len(training_input) - (number_of_fold_indexes * (k_folds - i - 1)))
        validation_x = training_input.drop(training_input.index[0:index_start])
        validation_x = validation_x.drop(validation_x.index[number_of_fold_indexes:len(validation_x)])
        validation_y = training_output.drop(training_output.index[0:index_start])
        validation_y = validation_y.drop(validation_y.index[number_of_fold_indexes:len(validation_y)])
        train_x = training_input.drop(training_input.index[index_start:index_end])
        train_y = training_output.drop(training_output.index[index_start:index_end])
        ridge.fit(X=train_x, y=train_y)
        test_values = ridge.predict(validation_x)
        train_values = ridge.predict(train_x)
        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))
    return np.mean(train_error), np.mean(cv_error)


# def predict(x, y, p):
#     ridge = Ridge(alpha=p)
#     ridge.fit(x, y)
#     prediction = ridge.predict(test_features)
#
#     prediction_error = np.mean(abs(prediction - test_label))
#     print("MAE at lamda: 10^", np.sqrt(p), prediction_error)
#
