import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import csv_reader as reader


# Radius classification, this is used to determine whether or not a claim is a claim or not and returns a dataframe
# Of claims
# Algorithm uses k fold cross validation.
# PARAM: training_x
# PARAM:
# PARAM:
# PARAM:
# RETURN:
def knn_classifier(claimed_x, claimed_y, r, k_folds):
    model = KNeighborsClassifier(n_neighbors=r)
    number_of_fold_indexes = int(len(claimed_x) / k_folds)
    train_error = []
    cv_error = []
    for i in range(k_folds):
        claim_train = []
        claim_validation = []
        index_start = int(i * number_of_fold_indexes)
        index_end = int(len(claimed_x) - (number_of_fold_indexes * (k_folds - i - 1)))
        validation_x = claimed_x.drop(claimed_x.index[0:index_start])
        validation_x = validation_x.drop(validation_x.index[number_of_fold_indexes:len(validation_x)])
        validation_y = claimed_y.drop(claimed_y.index[0:index_start])
        validation_y = validation_y.drop(validation_y.index[number_of_fold_indexes:len(validation_y)])
        train_x = claimed_x.drop(claimed_x.index[index_start:index_end])
        train_y = claimed_y.drop(claimed_y.index[index_start:index_end])

        for j in train_y:
            if j == 0:
                claim_train.append(0)
            elif j < 1000:
                claim_train.append(1)
            else:
                claim_train.append(2)
        for j in validation_y:
            if j == 0:
                claim_validation.append(0)
            elif j < 1000:
                claim_validation.append(1)
            else:
                claim_validation.append(2)
        print("Fitting model for k fold of K =", i)
        model.fit(X=train_x, y=claim_train)
        print("Predicting training set..")
        training_predictions = model.predict(train_x)
        print("Training model data:",
              claim_train.count(1),
              claim_train.count(2),
              np.unique(training_predictions, return_counts=True))
        print("Predicting CV set...")
        cv_predictions = model.predict(validation_x)
        print("CV Model data:",
              claim_validation.count(1),
              claim_validation.count(2),
              np.unique(cv_predictions, return_counts=True))
        train_error.append(((np.size(claim_train) - np.count_nonzero(training_predictions == claim_train)) / len(claim_train)))
        cv_error.append((np.size(claim_validation) - np.count_nonzero(cv_predictions == claim_validation) / len(claim_validation)))
    return np.mean(train_error), np.mean(cv_error)


def knn_filter(input_x, k_neighbors):
    training_x, training_y = reader.get_dataset_categorical()
    print(training_x.loc[[0]])
    print(input_x.loc[[0]])
    small_claims = pd.DataFrame(data=input_x)
    small_claims = small_claims.drop(small_claims.index[0:len(small_claims)])
    large_claims = pd.DataFrame(data=input_x)
    large_claims = large_claims.drop(large_claims.index[0:len(large_claims)])
    claim_count = []
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    for j in training_y:
        if j == 0:
            claim_count.append(0)
        elif j < 1000:
            claim_count.append(1)
        else:
            claim_count.append(2)
    print("Fitting KNN model...")
    model.fit(training_x, claim_count)
    print("Making KNN predictions...")
    predictions = model.predict(input_x)
    for i in range(len(predictions)):
        if predictions[i] == 1:
            small_claims = small_claims.append(input_x.loc[[i]])
        elif predictions[i] == 2:
            large_claims = large_claims.append(input_x.loc[[i]])
    return small_claims, large_claims, predictions
