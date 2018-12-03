import numpy as np
import pandas as pd
import csv_reader as reader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

'''
TODO: Implement RSS, and MAE, modify poly kfold to pass a function.
'''


# k-folds cross validation for polynomial regression
# PARAM: x - training input
# PARAM: y - training input
# PARAM: p - degree of the fitting polynomial
# PARAM: k - number of folds
# RETURN: train_error: Average MAE of training sets across all K folds.
# RETURN: cv_error: average MAE of the validation sets across all K folds.
# NOTES: train_error should return 1.0355, and cv_error should return 1.0848
def poly_reg_kfold_cv(training_input, training_output, p, k_folds):
    lin_reg = LinearRegression(fit_intercept=True, normalize=False)
    poly = PolynomialFeatures(degree=p)
    number_of_fold_indexes = int(len(training_input) / k_folds)
    train_error = []
    cv_error = []
    for i in range(k_folds):
        index_start = int(i * number_of_fold_indexes)
        index_end = int(len(training_input) - (number_of_fold_indexes * (k_folds - i - 1)))
        validation_x = training_input.drop(training_input.index[0:index_start])
        validation_x = validation_x.drop(validation_x.index[number_of_fold_indexes:len(validation_x)])
        validation_y = training_output.drop(training_output.index[0:index_start])
        validation_y = validation_y.drop(validation_y.index[number_of_fold_indexes:len(validation_y)])
        training_set_x = training_input.drop(training_input.index[index_start:index_end])
        training_set_y = training_output.drop(training_output.index[index_start:index_end])
        x_transform = poly.fit_transform(training_set_x)
        validation_transform = poly.fit_transform(validation_x)
        lin_reg.fit(x_transform, training_set_y)
        train_values = lin_reg.predict(x_transform)
        test_values = lin_reg.predict(validation_transform)
        for l in range(len(train_values)):
            train_error.append(abs(train_values[l] - training_set_y.tolist()[l]))
        for l in range(len(test_values)):
            cv_error.append(abs(test_values[l] - validation_y.tolist()[l]))
    return np.mean(train_error), np.mean(cv_error)


def ridge_kfold_cv(training_input, training_output, a, k_folds):
    train_error = []
    cv_error = []
    model = Ridge(alpha=a)
    number_of_fold_indexes = int(len(training_input) / k_folds)
    for i in range(k_folds):
        index_start = int(i * number_of_fold_indexes)
        index_end = int(len(training_input) - (number_of_fold_indexes * (k_folds - i - 1)))
        validation_x = training_input.drop(training_input.index[0:index_start])
        validation_x = validation_x.drop(validation_x.index[number_of_fold_indexes:len(validation_x)])
        validation_y = training_output.drop(training_output.index[0:index_start])
        validation_y = validation_y.drop(validation_y.index[number_of_fold_indexes:len(validation_y)])
        training_set_x = training_input.drop(training_input.index[index_start:index_end])
        training_set_y = training_output.drop(training_output.index[index_start:index_end])
        model.fit(training_set_x, training_set_y)
        cv_predict = model.predict(validation_x)
        train_predict = model.predict(training_set_x)
        train_error = (abs(train_predict - training_set_y))
        cv_error = (abs(cv_predict - validation_y))
    return np.mean(train_error), np.mean(cv_error)


def poly_reg_predict(input_x, training_input, training_output, degree):
    lin_reg = LinearRegression(fit_intercept=True, normalize=False)
    poly_reg = PolynomialFeatures(degree)
    x_transform = poly_reg.fit_transform(training_input)
    lin_reg.fit(x_transform, training_output)
    ploy_reg_input = PolynomialFeatures(degree)
    input_x_transform = ploy_reg_input.fit_transform(input_x)
    predictions = lin_reg.predict(X=input_x_transform)
    return predictions, lin_reg


def poly_reg_model_predict(input_x, model):
    return model.predict(input_x)
