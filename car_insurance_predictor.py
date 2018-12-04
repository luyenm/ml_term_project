import csv_reader as reader
import poly_reg as pg
import data_visualizer as dv
import best_subset_selector as bss
import kNN_classification as knn
import numpy as np
import tensorflow_regression as tf_reg
import scv_classification as svc
from sklearn import metrics as sklearn_metrics
import pandas as pd
import pickle as pk
from pathlib import Path

guesses = []

print("Getting datasets...")
x_claimed_money_categorical, y_claimed_money_categorical = reader.get_dataset_categorical()
x_test_set = reader.get_testset_categorical()

test_indices = int(len(x_claimed_money_categorical) * 0.7)
valid_indices = int(len(x_claimed_money_categorical - test_indices))

print("Splitting dataset into training and test set...")
test_x = x_claimed_money_categorical.loc[0:test_indices]
test_y = y_claimed_money_categorical.loc[0:test_indices]
valid_x = x_claimed_money_categorical.drop(x_claimed_money_categorical.index[test_indices:len(x_claimed_money_categorical)])
valid_y = y_claimed_money_categorical.drop(y_claimed_money_categorical.index[test_indices:len(x_claimed_money_categorical)])
kfolds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n_neighbors = [0.001, 0.01, 0.1, 1, 5, 10, 100]


# mae = []
# train_error = []
# cv_error = []
# for k in n_neighbors:
#     te, cve = svc.svc_kfold_cv(x_claimed_money_categorical, y_claimed_money_categorical, k, 5)
#     print("Training Error", te, "CV Error", cve, "C Value", k)
#     train_error.append(te)
#     cv_error.append(cve)
# dv.plot_line_graph(train_error, cv_error, n_neighbors, "Error", "Degrees", "Classification error graph")

# data = reader.get_trainset()
# split = int(data.shape[0]*0.7)
# train = data.loc[:split]
# test = data.loc[1 + split:]
# train_data = train.drop('ClaimAmount', axis=1, inplace=False)
# train_data = train_data.values
# train_labels = train.loc[:,'ClaimAmount']
# train_labels = train_labels.values
# test_data = test.drop('ClaimAmount', axis=1, inplace=False)
# test_data = test_data.values
# test_labels = test.loc[:,'ClaimAmount']
# test_labels = test_labels.values
# mae, history = tf_reg.adadelta_cv(tf_reg.build_Adadelta(train_data.shape[1]), train_data, train_labels, test_data, None, 500)
# print('mae', mae)

test_predictions = []

pickle_path = Path('saved_model.p')
competition_set_path = Path('competitionset.csv')
test_path = Path('testset.csv')
if not pickle_path.exists():
    print("Saved model not found, training...")
    print("Filtering Data...")
    prediction_set, claim_collection, knn_model = knn.knn_filter(x_claimed_money_categorical, 1)
    print(np.count_nonzero(claim_collection))
    print("Generating linear regression model...")
    predictions, poly_reg_model = pg.poly_reg_predict(prediction_set, x_claimed_money_categorical, y_claimed_money_categorical, 1)
    print("Pickling model...")
    model = {"knn": knn_model,
             "poly": poly_reg_model}

    saved_model = open("saved_model.p", "wb")
    pk.dump(model, saved_model)
    saved_model.close()
    print("Model trained, please run the program again.")
    exit(0)
else:
    print("Found saved model, predicting.")
    loaded_model = pk.load(open('saved_model.p', "rb"))
    kneighbors_model = loaded_model['knn']
    polynomial_model = loaded_model['poly']

    if competition_set_path.exists():
        print("Found competition set... predicting..")
        competition_set = reader.get_competitive_set_categorical()
        prediction_set, claim_collection = knn.knn_model_predict(competition_set, kneighbors_model)
        predictions = pg.poly_reg_model_predict(prediction_set, polynomial_model)

        print("Generating full list of predictions...")
        j = 0
        full_prediction = []
        for i in claim_collection:
            if i == 1:
                if predictions[j] <= 0:
                    predictions[j] = abs(predictions[j])
                full_prediction.append(predictions[j])
                j += 1
            else:
                full_prediction.append(0)

        print('Outputting CSV...')
        output = pd.DataFrame()
        rowIndex = range(len(full_prediction))
        output['rowIndex'] = rowIndex
        output['ClaimAmount'] = full_prediction
        output.to_csv('competition_predictions.csv', index=False)
    if test_path.exists():
        print("Found test set... predicting..")
        competition_set = reader.get_competitive_set_categorical()
        prediction_set, claim_collection = knn.knn_model_predict(competition_set, kneighbors_model)
        predictions = pg.poly_reg_model_predict(prediction_set, polynomial_model)
        print("Generating full list of predictions...")
        j = 0
        full_prediction = []
        for i in claim_collection:
            if i == 1:
                if predictions[j] <= 0:
                    predictions[j] = abs(predictions[j])
                full_prediction.append(predictions[j])
                j += 1
            else:
                full_prediction.append(0)
        print('Outputting CSV...')
        output = pd.DataFrame()
        rowIndex = range(len(full_prediction))
        output['rowIndex'] = rowIndex
        output['ClaimAmount'] = full_prediction
        output.to_csv('test_predictions.csv', index=False)

    prediction_set, claim_collection = knn.knn_model_predict(valid_x, kneighbors_model)
    predictions = pg.poly_reg_model_predict(prediction_set, polynomial_model)
    j = 0
    full_prediction = []
    for i in claim_collection:
        if i == 1:
            if predictions[j] <= 0:
                predictions[j] = abs(predictions[j])
            full_prediction.append(predictions[j])
            j += 1
        else:
            full_prediction.append(0)
    mae = np.mean(abs(full_prediction - valid_y))
    print("MAE: ", mae)


list_of_claims = []
for i in claim_collection:
    if i > 0:
        list_of_claims.append(1)
    else:
        list_of_claims.append(0)


# f1_score = sklearn_metrics.f1_score(prediction_y, predictions, average='macro')
