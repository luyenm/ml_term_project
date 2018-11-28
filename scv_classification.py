from sklearn.svm import SVC
import numpy as np
import csv_reader as reader
import pandas as pd


def svc_kfold_cv(input_data, output_data, alpha, kfolds=10):

    train_error = []
    cv_error = []
    model = SVC(kernel='linear', C=alpha)
    kfold_indices = int(len(input_data) / kfolds)
    for i in range(kfolds):
        claim_train = []
        claim_validation = []

        index_start = int(i * kfold_indices)
        index_end = int(len(input_data) - (kfold_indices * (kfolds - i - 1)))
        training_data = input_data.drop(input_data.index[index_start:index_end])
        training_output = output_data.drop(output_data.index[index_start:index_end])
        testing_data = input_data.loc[index_start:index_end]
        testing_output = output_data.loc[index_start:index_end]

        for j in training_output:
            if j != 0:
                claim_train.append(1)
            else:
                claim_train.append(0)
        for j in testing_output:
            if j != 0:
                claim_validation.append(1)
            else:
                claim_validation.append(0)

        print("Fitting Model...")

        model.fit(training_data, claim_train)

        print("Making Training Predictions...")
        training_predictions = model.predict(training_data)
        print("Making Cross Validation Predictions...")
        cv_predictions = model.predict(testing_data)

        train_error.append((np.size(training_predictions) - np.count_nonzero(training_predictions))
                           / np.size(training_predictions))
        cv_error.append((np.size(cv_predictions) - np.count_nonzero(cv_predictions))
                        / np.size(cv_predictions))

        return np.mean(train_error), np.mean(cv_error)


def svc_classifier(input_data, alpha):
    training_input, training_output = reader.get_dataset_categorical()
    filtered_x = pd.DataFrame(data=training_input)
    filtered_x = filtered_x.drop(filtered_x.index[0:len(filtered_x)])
    claim_count = []
    model = SVC(C=alpha, kernel='linear')
    for i in training_output:
        if i != 0:
            claim_count.append(1)
        else:
            claim_count.append(0)
    print("Fitting filter data...")
    model.fit(training_input, claim_count)
    print("Making predictions...")
    predictions = model.predict(input_data)
    print(np.count_nonzero(predictions))
    for i in range(len(input_data)):
        if predictions[i] == 1:
            filtered_x = filtered_x.append(input_data.loc[[i]])
    return filtered_x, predictions
