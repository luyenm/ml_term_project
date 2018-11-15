import pandas as pd
import numpy as np
import data_visualizer as dv
import os


'''
Current insights about the data:
ASSUMPTIONS:
Anything below 26 unique values is assume to be categorical.

# rowIndex - ID of person?
# feature1	(Values are between 0.001 and 1.000)
# feature2	(Values are between 9 and 940)
# feature3	(Values are 0 .. 8)
# feature4	(Values are 0, 1) - Categorical
# feature5	(Values are 0, 1) - Categorical
# feature6	(Values are between 0 and 51)
# feature7	(Values are 0 .. 3)
# feature8	(Values are between 18 and 103) Age?
# feature9	(Values are 0, 1) - Categorical
# feature10	(Values are 50 .. 272) Height?
# feature11	(Values are 0 .. 8) 
# feature12	(Values are 0 .. 26)
# feature13	(Values are 0 .. 5)
# feature14	(Values are 0 .. 3)
# feature15	(Values are 0 .. 9)
# feature16	(Values are 0 .. 5)
# feature17	(Values are 1 .. 20)
# feature18	(Values are 0, 1, 2)
# ClaimAmount - Dollar amount for claims that are made ($1 or $1000)
'''
# TODO: Implement more data pre-processing for categorical data.


def onehot(dataframe):
    raw_data = dataframe
    new_data = []
    for column in raw_data:
        if raw_data[column].nunique() <= 27:
            new_data.append(column)
    raw_data = pd.get_dummies(raw_data, columns=new_data, prefix=new_data)
    return raw_data


training_data = pd.read_csv('trainingset.csv')
training_data.drop('rowIndex', axis=1, inplace=True)

if not os.path.exists('scatterplot'):
    os.makedirs('scatterplot')
if not os.path.exists('histogram'):
    os.makedirs('histogram')

not_claimed = training_data[training_data.ClaimAmount == 0]
claimed_money = training_data[training_data.ClaimAmount != 0]

not_claimed_categorical = onehot(not_claimed)
claimed_money_categorical = onehot(claimed_money)

print('Not claim: ', len(not_claimed), '\t\tClaim Money: ', len(claimed_money))

x_claimed_money = claimed_money.drop('ClaimAmount', axis=1, inplace=False)
y_claimed_money = claimed_money.loc[:, 'ClaimAmount']
x_not_claimed = not_claimed.drop('ClaimAmount', axis=1, inplace=False)
y_not_claimed = not_claimed.loc[:, 'ClaimAmount']

x_claimed_money_categorical = claimed_money_categorical.drop('ClaimAmount', axis=1, inplace=False)
y_claimed_money_categorical = claimed_money_categorical.loc[:, 'ClaimAmount']
x_not_claimed_categorical = not_claimed.drop('ClaimAmount', axis=1, inplace=False)
y_not_claimed_categorical = not_claimed.loc[:, 'ClaimAmount']
