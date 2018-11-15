import csv_reader as reader
import poly_reg as pg
import data_visualizer as dv

x_claimed_money_categorical, y_claimed_money_categorical = reader.get_claims_categorical()

kfolds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
poly_reg = [1, 2, 3, 4, 4, 5, 6, 7]
train_error = []
cv_error = []
mae = []
for i in kfolds:
    for k in poly_reg:
        te, cve = pg.poly_kfold_cv(x_claimed_money_categorical, y_claimed_money_categorical, poly_reg, kfolds)
        train_error.append(te)
        cv_error.append(cve)
