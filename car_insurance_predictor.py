import csv_reader as reader
import poly_reg as pg
import data_visualizer as dv
import best_subset_selector as bss
import kNN_classification as knn
import numpy as np
import tensorflow_regression as tf_reg

x_claimed_money_categorical, y_claimed_money_categorical = reader.get_dataset_categorical()
x_test_set = reader.get_testset_categorical()

test_indices = int(len(x_claimed_money_categorical) * 0.7)
valid_indices = int(len(x_claimed_money_categorical - test_indices))

#test_x = x_claimed_money_categorical.loc[0:test_indices]
#test_y = y_claimed_money_categorical.loc[0:test_indices]
#valid_x = x_claimed_money_categorical.drop(x_claimed_money_categorical.index[test_indices:len(x_claimed_money_categorical)])
#valid_y = y_claimed_money_categorical.drop(y_claimed_money_categorical.index[test_indices:len(x_claimed_money_categorical)])
kfolds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n_neighbors = range(0, 5, 1)

# mae = []
# train_error = []
# cv_error = []
# for k in n_neighbors:
#     te, cve = pg.ridge_kfold_cv(x_claimed_money_categorical, y_claimed_money_categorical, 10**k, 10)
#     print("Training Error", te, "CV Error", cve, "K neighbors:", k)
#     train_error.append(te)
#     cv_error.append(cve)
# dv.plot_line_graph(train_error, cv_error, n_neighbors, "Error", "Degrees", "Classification error graph")

data = reader.get_trainset()
split = int(data.shape[0]*0.7)
train = data.loc[:split]
test = data.loc[1 + split:]
train_data = train.drop('ClaimAmount', axis=1, inplace=False)
train_data = train_data.values
train_labels = train.loc[:,'ClaimAmount']
train_labels = train_labels.values
test_data = test.drop('ClaimAmount', axis=1, inplace=False)
test_data = test_data.values
test_labels = test.loc[:,'ClaimAmount']
test_labels = test_labels.values
mae, history = tf_reg.adadelta_cv(tf_reg.build_Adadelta(train_data.shape[1]), train_data, train_labels, test_data, None, 500)
print('mae', mae)



# prediction_set, claim_collection = knn.knn_filter(x_test_set, 1)
# predictions = pg.poly_reg_predict(x_test_set, x_claimed_money_categorical, y_claimed_money_categorical, 1)
# j = 0
# for i in claim_collection:
#     if i == 1:
#         if predictions[j] < 0:
#             predictions[j] = 0
#         claim_collection[i] = predictions[j]
#         j += 1
#     print(claim_collection[i])
# print(len(predictions), len(valid_y))
#
#
# print(np.mean(abs(claim_collection-valid_y)))

