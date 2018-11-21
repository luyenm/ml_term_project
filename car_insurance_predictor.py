import csv_reader as reader
import poly_reg as pg
import data_visualizer as dv
import best_subset_selector as bss
import kNN_classification as knn

x_claimed_money_categorical, y_claimed_money_categorical = reader.get_dataset()
test_set = reader.get_testset()

kfolds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
poly_reg = range(1, 15, 1)

mae = []
train_error = []
cv_error = []
# for k in poly_reg:
#     te, cve = knn.knn_classifier(x_claimed_money_categorical, y_claimed_money_categorical, k, 5)
#     print("Training Error", te, "CV Error", cve, "K neighbors:", k)
#     train_error.append(te)
#     cv_error.append(cve)
# dv.plot_line_graph(train_error, cv_error, poly_reg, "Error", "Degrees", "Error for ridge regression degrees")

training_set = knn.knn_filter(test_set, 15)
print(len(test_set), len(training_set))

