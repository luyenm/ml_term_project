import csv_reader as reader
import numpy as np
from sklearn.linear_model import Ridge

# TODO: Implement forward-stepwise subset selection.
# TODO: Implement ridge regression for the 110 columns after data pre-processing


# Performs ridge regression and K-Fold cross validation.
# x - inputs
# y - outputs
# k - kfolds
# p - lambda
def ridge_regression(x, y, k, p):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    ridge = Ridge(alpha=p)
    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        train_y = y[index_start:index_end]
        train_x = x[index_start:index_end]
        validation_x = x.drop(x.index[index_start:index_end])
        validation_y = y.drop(y.index[index_start:index_end])
        ridge.fit(X=train_x, y=train_y)
        test_values = ridge.predict(validation_x)
        train_values = ridge.predict(train_x)
        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))
    return np.mean(train_error), np.mean(cv_error)
