import numpy as np
import pandas as pd
import csv_reader as reader


# k-folds cross validation for polynomial regression
# PARAM: x - training input
# PARAM: y - training input
# PARAM: p - degree of the fitting polynomial
# PARAM: k - number of folds
# RETURN: train_error: Average MAE of training sets across all K folds.
# RETURN: cv_error: average MAE of the validation sets across all K folds.
# NOTES: train_error should return 1.0355, and cv_error should return 1.0848
def poly_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        validation_y = y[index_start:index_end]
        validation_x = x[index_start:index_end]
        train_x = x.drop(x.index[index_start:index_end])
        train_y = y.drop(y.index[index_start:index_end])
        test_coefficient = np.polyfit(x=train_x, y=train_y, deg=p)
        train_values = np.polyval(test_coefficient, train_x)
        test_values = np.polyval(test_coefficient, validation_x)
        for l in range(len(train_values)):
            train_error.append(abs(train_values[l] - train_y.tolist()[l]))

        for l in range(len(test_values)):
            cv_error.append(abs(test_values[l] - validation_y.tolist()[l]))
    return np.mean(train_error), np.mean(cv_error)