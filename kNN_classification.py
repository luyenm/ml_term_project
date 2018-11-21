import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


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

        training_incorrect = 0
        cv_incorrect = 0
        index_start = int(i * number_of_fold_indexes)
        index_end = int(len(claimed_x) - (number_of_fold_indexes * (k_folds - i - 1)))
        validation_x = claimed_x.drop(claimed_x.index[0:index_start])
        validation_x = validation_x.drop(validation_x.index[number_of_fold_indexes:len(validation_x)])
        validation_y = claimed_y.drop(claimed_y.index[0:index_start])
        validation_y = validation_y.drop(validation_y.index[number_of_fold_indexes:len(validation_y)])
        train_x = claimed_x.drop(claimed_x.index[index_start:index_end])
        train_y = claimed_y.drop(claimed_y.index[index_start:index_end])

        for j in train_y:
            if j != 0:
                claim_train.append(1)
            else:
                claim_train.append(0)
        for j in validation_y:
            if j != 0:
                claim_validation.append(1)
            else:
                claim_validation.append(0)

        model.fit(X=train_x, y=claim_train)
        training_predictions = model.predict(train_x)
        cv_predictions = model.predict(validation_x)
        for j in range(len(training_predictions)):
            if training_predictions[j] != claim_train[j]:
                training_incorrect += 1
        for j in range(len(cv_predictions)):
            # print("Cross validation predicton:", cv_predictions[j], "Actual Prediction: ", validation_y.tolist()[j])
            if cv_predictions[j] != claim_validation[j]:
                cv_incorrect += 1
        train_error.append(training_incorrect/len(training_predictions))
        cv_error.append(cv_incorrect/len(cv_predictions))

    return np.mean(train_error), np.mean(cv_error)
