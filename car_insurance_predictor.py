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

n_neighbors = [1]


def test_model(input_values, output_values):
    mae = []
    train_error = []
    cv_error = []
    for k in n_neighbors:
        te, cve = knn.knn_classifier(input_values, output_values, k, 5)
        print("Training Error", te, "CV Error", cve, "C Value", k)
        train_error.append(te)
        cv_error.append(cve)
    print("Model assessment MAE:", np.mean(cv_error))


test_model(test_x, test_y)


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
small_claims, large_claims, claim_collection = knn.knn_filter(valid_x, 1)
print(np.count_nonzero(claim_collection))
print("Predicting...")
print(len(small_claims), len(large_claims))
# predictions = tf_reg.adadelta_cv(tf_reg.build_Adadelta(x_claimed_money_categorical.shape[1]), x_claimed_money_categorical, y_claimed_money_categorical, prediction_set, None, 100)
small_predictions = pg.poly_reg_predict(small_claims, test_x, test_y, 1)
large_predictions = pg.poly_reg_predict(large_claims, test_x, test_y, 1)
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
k = 0
full_prediction = []
for i in claim_collection:
    if i == 1 and small_predictions[j] > 0:
        full_prediction.append(small_predictions[j])
        j += 1
    elif i == 2 and large_predictions[k] > 0:
        full_prediction.append(large_predictions[k])
        k += 1
    else:
        full_prediction.append(0)
# print('f1_score', f1_score)
# print(len(predictions), len(valid_y))

# for i in full_prediction:
#     print(i)

print('Outputting CSV...')
output = pd.DataFrame()
rowIndex = range(len(full_prediction))
output['rowIndex'] = rowIndex
# output = pd.DataFrame(full_prediction, columns=['ClaimAmount'])
output['ClaimAmount'] = full_prediction
output.to_csv('predictedclaimamount.csv', index=False)

print(np.mean(abs(full_prediction - valid_y)))
#
# for i in range(len(claim_collection)):
#     print('{:7.2f}'.format(full_prediction[i]), "\t\t\t", valid_y[i])
