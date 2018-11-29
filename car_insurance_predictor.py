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

print(np.count_nonzero(valid_y))
print("Filtering Data...")
prediction_set, claim_collection = knn.knn_filter(valid_x, valid_y, 1)
print(np.count_nonzero(claim_collection))
print("Predicting...")
# predictions = tf_reg.adadelta_cv(tf_reg.build_Adadelta(x_claimed_money_categorical.shape[1]), x_claimed_money_categorical, y_claimed_money_categorical, prediction_set, None, 100)
predictions = pg.poly_reg_predict(prediction_set, test_x, test_y, 1)

print("Generating a list of claims for F1 score...")
list_of_claims = []
for i in claim_collection:
    if i > 0:
        list_of_claims.append(1)
    else:
        list_of_claims.append(0)


# f1_score = sklearn_metrics.f1_score(prediction_y, predictions, average='macro')

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
# print('f1_score', f1_score)
print(np.count_nonzero(full_prediction), np.count_nonzero(valid_y))
print(np.mean(abs(full_prediction - valid_y)))

# for i in full_prediction:
#     print(i)

print('Outputting CSV...')
output = pd.DataFrame()
rowIndex = range(len(full_prediction))
output['rowIndex'] = rowIndex
# output = pd.DataFrame(full_prediction, columns=['ClaimAmount'])
output['ClaimAmount'] = full_prediction
output.to_csv('predictedclaimamount.csv', index=False)

#
# for i in range(len(claim_collection)):
#     print('{:7.2f}'.format(full_prediction[i]), "\t\t\t", valid_y[i])
